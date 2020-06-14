import io
import logging
import os
import audioop
import wave
from enum import Enum, auto
from struct import Struct

import dialogflow_v2.types

from discord.ext import commands
from discord.opus import Decoder, Encoder, _load_default
from discord.reader import AudioSink
from discord.player import AudioSource
from gtts import gTTS
from pvporcupine import create
from pydub import AudioSegment
from serpapi.google_search_results import GoogleSearchResults

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class State(Enum):
    WaitingWuw = auto()
    WaitingQuestion = auto()
    ListeningQuestion = auto()


class PorcupineSink(AudioSink, AudioSource):
    def __init__(self, voice_client, keywords):
        self.pcp = create(keywords=keywords)
        self.audio_state = None
        self.input = b''
        self.raw_input = b''
        self.to_skip = 0
        self.unpacker = Struct(f'{self.pcp.frame_length}h')
        self.output = b''
        self.what = wave.open('what2.wav', 'rb').readframes(100000)
        self.that = wave.open('that2.wav', 'rb').readframes(100000)
        self.state = State.WaitingWuw
        self.play(self.what)
        self.play(self.that)
        self.deleted = False

        self.dialogflow_client = dialogflow_v2.SessionsClient()
        self.dialogflow_project_id = 'agent-322-kolya-hosjfw'

        voice_client.listen(self)
        voice_client.play(self, after=lambda e: print('Player error: %s' % e) if e else None)

        logger.debug('Started Porcupine sink')

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
            logger.debug(f'last second rms: {rms}')
            if rms == 0:
                logger.debug('Detected silence')
                self.play(self.that)
                mono = audioop.tomono(self.raw_input, width, 0.5, 0.5)
                self.raw_input = b''
                self.detect_intent(mono, data.user)
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
                if result:
                    logger.debug('Detected keyword')
                    self.play(self.what)
                    self.state = State.WaitingQuestion
                    self.to_skip = len(self.what)
                    self.input = b''
                else:
                    logger.debug(f'last frame rms: {rms}')

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
        if response.query_result.action == 'google.search':
            self.google_search(response.query_result.parameters.fields['query'].string_value)
        if response.query_result.all_required_params_present:
            self.state = State.WaitingWuw
        else:
            self.state = State.WaitingQuestion
            self.to_skip = len(answer)

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
        logger.debug(f'serp results: {results}')
        if 'knowledge_graph' in results:
            answer = results['knowledge_graph']['description']
            self.play(self.text_to_speech(answer))
        else:
            self.play(self.text_to_speech("что-то я не нашла"))
        self.state = State.WaitingWuw

    def text_to_speech(self, text):
        data = io.BytesIO()
        gTTS(text, lang='ru').write_to_fp(data)
        data.seek(0)
        segment = AudioSegment.from_mp3(data)
        logger.debug(f'TTS result: {segment.sample_width}, {segment.channels}, {segment.frame_rate}')
        converted, _ = audioop.ratecv(
            segment.raw_data, segment.sample_width, segment.channels, segment.frame_rate, Encoder.SAMPLING_RATE, None,
        )
        return audioop.tostereo(converted, segment.sample_width, 1, 1)

    def cleanup(self):
        if self.deleted:
            return
        self.deleted = True
        logger.debug('Deleting Porcupine')
        self.pcp.delete()


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
        PorcupineSink(ctx.voice_client, keywords=[keyword])

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
    PorcupineSink(voice_client, ['computer'])

bot.add_cog(DialogflowCog(bot))
bot.run(os.getenv('TOKEN'))
