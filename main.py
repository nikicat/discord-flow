import io
import logging
import os
import audioop
import wave
from enum import Enum, auto
from struct import Struct
from threading import Thread
from queue import Queue

import dialogflow_v2.types
import requests

from discord.ext import commands
from discord.opus import Decoder, Encoder, _load_default
from discord.reader import AudioSink
from discord.player import AudioSource
from google.cloud import texttospeech
from pvporcupine import create
from pydub import AudioSegment
from serpapi.google_search_results import GoogleSearchResults

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class State(Enum):
    WaitingWuw = auto()
    WaitingQuestion = auto()
    ListeningQuestion = auto()
    Answering = auto()


class PorcupineSink:
    def __init__(self, parent, keywords):
        self.parent = parent
        self.what = parent.what
        self.that = parent.that
        self.pcp = create(keywords=keywords, sensitivities=[0.5] * len(keywords))
        self.audio_state = None
        self.input = b''
        self.raw_input = b''
        self.to_skip = 0
        self.unpacker = Struct(f'{self.pcp.frame_length}h')
        self.state = State.WaitingWuw
        self.voice = texttospeech.VoiceSelectionParams(
            language_code='ru_RU', ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )

        self.dialogflow_client = dialogflow_v2.SessionsClient()
        self.dialogflow_project_id = 'agent-322-kolya-hosjfw'
        self.tasks = Queue()
        self.back_thread = Thread(target=self.back_loop)
        self.back_thread.daemon = True
        self.back_thread.start()

        logger.debug('Started Porcupine sink')

    def back_loop(self):
        while True:
            data, user = self.tasks.get()
            self.detect_intent(data, user)
            self.tasks.task_done()

    def play(self, data):
        self.parent.play(data)

    def write(self, data):
        width = Decoder.SAMPLE_SIZE // Decoder.CHANNELS
        rms = AudioSegment(
            data=data.data,
            sample_width=width, frame_rate=Decoder.SAMPLING_RATE, channels=Decoder.CHANNELS,
        ).rms
        if self.state is State.WaitingQuestion:
            if self.to_skip > 0:
                self.to_skip -= len(data.data)
            elif rms > 0:
                self.state = State.ListeningQuestion
                logger.debug('Start listening')
        elif self.state is State.ListeningQuestion:
            self.raw_input += data.data
            silence_length_seconds = 0.5
            silence_length_bytes = int(silence_length_seconds * Decoder.SAMPLING_RATE * Decoder.SAMPLE_SIZE)
            if len(self.raw_input) < silence_length_bytes:
                return
            rms = AudioSegment(
                data=self.raw_input[len(self.raw_input) - silence_length_bytes:],
                sample_width=width, frame_rate=Decoder.SAMPLING_RATE, channels=Decoder.CHANNELS,
            ).rms
            if rms == 0:
                logger.debug('Detected silence')
                self.play(self.that)
                mono = audioop.tomono(self.raw_input, width, 0.5, 0.5)
                self.raw_input = b''
                self.tasks.put((mono, data.user))
                self.state = State.Answering
        elif self.state is State.WaitingWuw:
            mono = audioop.tomono(data.data, width, 0.5, 0.5)
            converted, self.audio_state = audioop.ratecv(
                mono, width, 1, Decoder.SAMPLING_RATE, self.pcp.sample_rate, self.audio_state,
            )
            self.input += converted
            while len(self.input) >= self.pcp.frame_length * width:
                to_process = self.input[:self.pcp.frame_length * width]
                self.input = self.input[self.pcp.frame_length * width:]
                result = self.pcp.process(self.unpacker.unpack(to_process))
                if result >= 0:
                    logger.debug(f'Detected keyword {result}')
                    self.play(self.what)
                    self.state = State.WaitingQuestion
                    self.to_skip = len(self.what)
                    self.input = b''
        elif self.state is State.Answering:
            pass
        #logger.debug(f'{data.user} {self.state} {rms}')

    def detect_intent(self, data, user):
        session = self.dialogflow_client.session_path(self.dialogflow_project_id, user)
        logger.debug(f"Session: {session}")
        query_input = dialogflow_v2.types.QueryInput(audio_config=dialogflow_v2.types.InputAudioConfig(
            audio_encoding=self.dialogflow_client.enums.AudioEncoding.AUDIO_ENCODING_LINEAR_16,
            sample_rate_hertz=Decoder.SAMPLING_RATE,
            language_code='ru',
            enable_word_info=True,
        ))
        response = self.dialogflow_client.detect_intent(
            session=session,
            query_input=query_input,
            output_audio_config=dialogflow_v2.types.OutputAudioConfig(
                audio_encoding=self.dialogflow_client.enums.AudioEncoding.AUDIO_ENCODING_LINEAR_16,
                sample_rate_hertz=Encoder.SAMPLING_RATE,
            ),
            input_audio=data,
        )
        if response.output_audio:
            answer = wave.open(io.BytesIO(response.output_audio), 'rb').readframes(999999999)
            mono_answer = audioop.tostereo(answer, 2, 1, 1)
            self.play(mono_answer)
        else:
            answer = b''
        action = response.query_result.action
        parameters = response.query_result.parameters.fields
        query = parameters['query'].string_value if 'query' in parameters else response.query_result.query_text
        if action == 'google.search':
            answer = self.google_search(query)
        elif action == 'google.news':
            answer = self.google_news(parameters['subject'].string_value)
        elif action == 'format':
            answer = response.query_result.filfillment_text.format(query=query)
        elif action == 'pbot':
            answer = self.query_pbot(query)
        else:
            answer = None
        if answer:
            self.play(self.text_to_speech(answer))
            self.state = State.WaitingWuw
        if response.query_result.all_required_params_present:
            self.state = State.WaitingWuw
        elif response.query_result.query_text:
            self.state = State.WaitingQuestion
            self.to_skip = len(answer)
        else:
            self.state = State.WaitingWuw

        response.output_audio = b''
        print(response)
        print(response.query_result.query_text)
        print(response.query_result.fulfillment_text)

    def google_search(self, query):
        params = dict(
            engine='google',
            q=query,
            api_key=os.getenv('SERPAPI_KEY'),
            gl='ru',
            hl='ru',
        )

        client = GoogleSearchResults(params)
        results = client.get_dict()
        answer = None
        logger.debug(f'serp results: {results}')
        if 'knowledge_graph' in results:
            knowledge_graph = results['knowledge_graph']
            if 'description' in knowledge_graph:
                answer = knowledge_graph['description']
                logger.debug(f'knowledge graph: {answer}')
        if answer is None and 'answer_box' in results:
            answer_box = results['answer_box']
            if 'result' in answer_box:
                answer = answer_box['result']
                logger.debug(f'answer box result: {answer}')
            elif 'snippet' in answer_box:
                answer = answer_box['snippet']
                logger.debug(f'answer box snippet: {answer}')
        if answer is None and results.get('organic_results'):
            answer = results['organic_results'][0].get('snippet')
        return answer or "что-то я не нашла"

    def google_news(self, subject):
        params = dict(
            tbm='nws',
            q=subject,
            api_key=os.getenv('SERPAPI_KEY'),
            gl='ru',
            hl='ru',
        )

        client = GoogleSearchResults(params)
        results = client.get_dict()
        logger.debug(f'news results: {results}')
        answer = results.get('news_results',  [{}])[0].get('snippet')
        return answer or "что-то я не нашла"

    def query_pbot(self, query):
        data = dict(
            request=query,
            dialog_lang='ru',
            dialog_greeting=False,
            a='public-api',
            b=2344965608,
            c='3583983984',
            d='3846106130',
            e='0.3083590588085976',
            t='1592231233017',
            x='3.795646366191572',
        )
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
        }
        url = 'http://p-bot.ru/api/getAnswer'
        response = requests.post(url, data=data, headers=headers)
        if response.status_code != 200:
            logger.debug(f'pBot response: {response}: {response.text}')
            return "что-то ничего в голову не приходит"
        return response.json()['answer']

    def text_to_speech(self, text):
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
        response = client.synthesize_speech(input=synthesis_input, voice=self.voice, audio_config=audio_config)
        data = io.BytesIO(response.audio_content)
        audio = wave.open(data, 'rb')
        converted, _ = audioop.ratecv(
            audio.readframes(9999999), audio.getsampwidth(), audio.getnchannels(), audio.getframerate(), Encoder.SAMPLING_RATE,
            None,
        )
        return audioop.tostereo(converted, audio.getsampwidth(), 1, 1)

    def close(self):
        self.pcp.delete()


class DemultiplexerSink(AudioSink, AudioSource):
    def __init__(self, voice_client, keywords):
        self.keywords = keywords
        self.users = {}
        self.output = b''
        self.deleted = False
        self.what = wave.open('what2.wav', 'rb').readframes(100000)
        self.that = wave.open('that2.wav', 'rb').readframes(100000)
        self.play(self.what)
        self.play(self.that)

        voice_client.listen(self)
        voice_client.play(self, after=lambda e: print('Player error: %s' % e) if e else None)

        logger.debug('Started Porcupine sink')

    def write(self, data):
        if data.user in self.users:
            usersink = self.users[data.user]
        else:
            self.users[data.user] = usersink = PorcupineSink(self, self.keywords)
        usersink.write(data)

    def read(self):
        result = b''
        if self.output:
            result += self.output[:Encoder.FRAME_SIZE]
            self.output = self.output[Encoder.FRAME_SIZE:]
        if len(result) < Encoder.FRAME_SIZE:
            result += b'\x00' * (Encoder.FRAME_SIZE - len(result))
        return result

    def play(self, data: bytes):
        self.output += data

    def cleanup(self):
        if self.deleted:
            return
        self.deleted = True
        logger.debug('Deleting Porcupine')
        for usersink in self.users.values():
            usersink.close()


class DialogflowCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        logging.basicConfig()

    @commands.command()
    async def join(self, ctx):
        """Joins a voice channel"""

        if ctx.voice_client.channel != ctx.author.voice.channel:
            return await ctx.voice_client.move_to(ctx.author.voice.channel)

    @commands.command()
    async def stop(self, ctx):
        """Stops and disconnects the bot from voice"""

        await ctx.voice_client.disconnect()

    @commands.command()
    async def listen(self, ctx, keyword: str):
        DemultiplexerSink(ctx.voice_client, keywords=[keyword])

    @join.before_invoke
    @listen.before_invoke
    async def ensure_voice(self, ctx):
        if ctx.voice_client is None:
            if ctx.author.voice:
                logger.debug(f'Joining channel {ctx.author.voice.channel!r}')
                await ctx.author.voice.channel.connect()
            else:
                await ctx.send("You are not connected to a voice channel.")
                raise commands.CommandError("Author not connected to a voice channel.")
        elif ctx.voice_client.is_playing():
            ctx.voice_client.stop()


bot = commands.Bot(command_prefix=commands.when_mentioned_or("!"),
                   description='Relatively simple music bot example')


@bot.event
async def on_ready():
    print('Logged in as {0} ({0.id})'.format(bot.user))
    print('------')
    _load_default()
    channel = bot.get_channel(653229977545998350)
    voice_client = await channel.connect()
    DemultiplexerSink(voice_client, [
        'view glass',
        'grasshopper',
        'blueberry',
        'jarvis',
        'bumblebee',
        'porcupine',
        'picovoice',
        'snowboy',
        'grapefruit',
        'smart mirror',
        'alexa',
        'americano',
        'computer',
        'terminator',
    ])

bot.add_cog(DialogflowCog(bot))
bot.run(os.getenv('TOKEN'))
