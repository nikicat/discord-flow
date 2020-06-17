import asyncio
import audioop
import io
import logging
import os
import time
import wave
from contextlib import suppress
from functools import partial
from itertools import count
from struct import Struct

import dialogflow_v2.types
from discord.ext import commands
from discord.opus import Decoder, Encoder, _load_default
from discord.reader import AudioSink
from google.cloud import texttospeech
from pvporcupine import create
from serpapi.google_search_results import GoogleSearchResults

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


SAMPLE_WIDTH = Decoder.SAMPLE_SIZE // Decoder.CHANNELS


class Interrupted(Exception):
    pass


class BackgroundTask:
    def __init__(self):
        self.task = None

    def start(self, coro):
        if self.task is not None:
            raise RuntimeError("{self!r} is already running")
        self.task = asyncio.create_task(coro)

    async def stop(self):
        self.task.cancel()
        with suppress(asyncio.CancelledError):
            await self.task
        self.task = None


class Waiter(BackgroundTask):
    def set(self, delay, callback):
        self.start(self.wait(delay, callback))

    async def wait(self, delay, callback):
        await asyncio.sleep(delay)
        await callback()


async def sync_to_async(func, *args, **kwargs):
    return await asyncio.get_running_loop().run_in_executor(None, partial(func, *args, **kwargs))


class PorcupineSink:
    def __init__(self, parent, keywords, user):
        self.parent = parent
        self.user = user
        self.pcp = create(keywords=keywords, sensitivities=[0.5] * len(keywords))
        self.audio_state = None
        self.input = b''
        self.question_input = b''
        self.unpacker = Struct(f'{self.pcp.frame_length}h')
        self.voice = texttospeech.VoiceSelectionParams(
            language_code='ru_RU', ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
        self.input_queue = asyncio.Queue()

        self.dialogflow_client = dialogflow_v2.SessionsClient()
        self.dialogflow_project_id = 'agent-322-kolya-hosjfw'
        self.background_player = BackgroundTask()
        self.skill_task = BackgroundTask()

        self.what = parent.what
        self.that = parent.that
        self.ticktock = parent.ticktock
        self.keywords = parent.keywords
        self.play = parent.play
        self.play_loop = parent.play_loop

        self.listen_task = BackgroundTask()
        self.listen_task.start(self.main_loop())

        logger.debug(f'Started Porcupine sink {self!r}')

    async def utterance_ended(self):
        await self.background_player.stop()
        utterance = audioop.tomono(self.question_input, SAMPLE_WIDTH, 0.5, 0.5)
        self.question_input = b''
        self.skill_task.start(self.process_utterance(utterance))

    def detect_wuw(self, sound):
        mono = audioop.tomono(sound, SAMPLE_WIDTH, 0.5, 0.5)
        converted, _ = audioop.ratecv(
            mono, SAMPLE_WIDTH, 1, Decoder.SAMPLING_RATE, self.pcp.sample_rate, None,
        )
        result = self.pcp.process(self.unpacker.unpack(converted))
        if result >= 0:
            return self.keywords[result]

    async def main_loop(self):
        while True:
            try:
                await self.wait_for_wuw()
                utterance = await self.listen_utterance()
                await self.process_utterance(utterance)
            except Interrupted:
                pass
            except Exception as exc:
                logger.exception(f"Exception in main loop: {exc}")

    async def wait_for_wuw(self):
        sound = b''
        pcp_frame_size = int(self.pcp.frame_length / self.pcp.sample_rate * Decoder.SAMPLING_RATE * Decoder.SAMPLE_SIZE)
        while True:
            sound += await self.listen()
            while len(sound) >= pcp_frame_size:
                to_process = sound[:pcp_frame_size]
                sound = sound[pcp_frame_size:]
                keyword = self.detect_wuw(to_process)
                if keyword:
                    logger.debug(f'Detected keyword "{keyword}"')
                    return keyword

    async def listen_utterance(self):
        logger.debug("Listening utterance")
        self.background_player.start(self.play_loop(self.ticktock))
        sound = b''
        while True:
            try:
                sound += await asyncio.wait_for(self.listen(), timeout=0.4)
            except asyncio.TimeoutError:
                await self.background_player.stop()
                return sound

    async def listen(self):
        return await self.input_queue.get()

    def write(self, data):
        self.input_queue.put_nowait(data.pcm)
        logger.debug(f'write: {data.user}. qsize={self.input_queue.qsize()}')
        return

    async def play_answer(self, answer):
        logger.debug("Playing answer")
        wait_for_wuw = self.wait_for_wuw()
        play = self.play(answer)
        done, pending = await asyncio.wait([wait_for_wuw, play], return_when=asyncio.FIRST_COMPLETED)
        pending = pending.pop()
        pending.cancel()
        with suppress(asyncio.CancelledError):
            await pending
        if pending.get_coro() is play:
            logger.debug("Interrupting")
            raise Interrupted

    async def process_utterance(self, utterance):
        session = self.dialogflow_client.session_path(self.dialogflow_project_id, self.user)
        logger.debug(f"Session: {session}")
        query_input = dialogflow_v2.types.QueryInput(audio_config=dialogflow_v2.types.InputAudioConfig(
            audio_encoding=self.dialogflow_client.enums.AudioEncoding.AUDIO_ENCODING_LINEAR_16,
            sample_rate_hertz=Decoder.SAMPLING_RATE,
            language_code='ru',
            enable_word_info=True,
        ))

        response = await sync_to_async(
            self.dialogflow_client.detect_intent,
            session=session,
            query_input=query_input,
            output_audio_config=dialogflow_v2.types.OutputAudioConfig(
                audio_encoding=self.dialogflow_client.enums.AudioEncoding.AUDIO_ENCODING_LINEAR_16,
                sample_rate_hertz=Encoder.SAMPLING_RATE,
            ),
            input_audio=audioop.tomono(utterance, SAMPLE_WIDTH, 0.5, 0.5),
        )
        output_audio = response.output_audio
        response.output_audio = b''
        logger.debug(f"Response: {response}")
        logger.debug(f"Query: {response.query_result.query_text}")
        logger.debug(f"Fulfillment: {response.query_result.fulfillment_text}")
        if output_audio:
            answer = wave.open(io.BytesIO(output_audio), 'rb').readframes(999999999)
            mono_answer = audioop.tostereo(answer, 2, 1, 1)
            await self.play_answer(mono_answer)
        action = response.query_result.action
        parameters = response.query_result.parameters.fields
        query = parameters['query'].string_value if 'query' in parameters else response.query_result.query_text
        if output_audio and not response.query_result.all_required_params_present:
            user_answer = await self.listen_utterance()
            await self.process_utterance(user_answer)
        elif action == 'google.search':
            answer = await self.google_search(query)
            await self.play_answer(await self.text_to_speech(answer))
        elif action == 'format':
            answer = response.query_result.filfillment_text.format(query=query)
            await self.play_answer(answer)
        elif action == 'skill':
            skill = parameters['skill'].string_value
            params = {p: parameters[p].string_value for p in parameters}
            await registry.run_skill(self, skill, params)
        elif action is None and response.output_audio is None:
            logger.debug("Empty response")

    async def ask_question(self, question):
        await self.play_answer(await self.text_to_speech(question))
        return await self.listen_utterance()

    async def google_search(self, query):
        params = dict(
            engine='google',
            q=query,
            api_key=os.getenv('SERPAPI_KEY'),
            gl='ru',
            hl='ru',
        )

        client = await sync_to_async(GoogleSearchResults, params)
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

    async def speak(self, text):
        audio = await self.text_to_speech(text)
        await self.play_answer(audio)

    def ddg_search(self, query):
        pass

    async def text_to_speech(self, text):
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=text)
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16)
        response = await sync_to_async(
            client.synthesize_speech, input=synthesis_input, voice=self.voice, audio_config=audio_config,
        )
        data = io.BytesIO(response.audio_content)
        audio = wave.open(data, 'rb')
        converted, _ = audioop.ratecv(
            audio.readframes(9999999), audio.getsampwidth(), audio.getnchannels(), audio.getframerate(), Encoder.SAMPLING_RATE,
            None,
        )
        return audioop.tostereo(converted, audio.getsampwidth(), 1, 1)

    async def close(self):
        await self.listen_task.stop()
        self.pcp.delete()


def load_sound(filename):
    return wave.open(filename, 'rb').readframes(100000)


class DemultiplexerSink(AudioSink):
    def __init__(self, voice_client, keywords):
        self.client = voice_client
        self.keywords = keywords
        self.users = {}
        self.deleted = False
        self.what = load_sound('what2.wav')
        self.that = load_sound('that2.wav')
        self.hello = load_sound('hello.wav')
        self.ticktock = load_sound('ticktock.wav')
        voice_client.listen(self)
        logger.debug('Started Porcupine sink')

    def write(self, data):
        if data.user not in self.users:
            self.users[data.user] = PorcupineSink(self, self.keywords, data.user)
        self.users[data.user].write(data)

    def read(self):
        result = b''
        if self.output:
            result += self.output[:Encoder.FRAME_SIZE]
            self.output = self.output[Encoder.FRAME_SIZE:]
        elif self.loop_output:
            start = self.loop_position
            stop = start + Encoder.FrameSize
            result = self.loop_output[start:stop]
            self.loop_position = stop if stop < len(self.loop_output) else 0
        if len(result) < Encoder.FRAME_SIZE:
            result += b'\x00' * (Encoder.FRAME_SIZE - len(result))
        return result

    async def play(self, sound: bytes):
        when_to_wake = time.perf_counter()
        for frame_number in count():
            offset = frame_number * Encoder.FRAME_SIZE
            frame = sound[offset:offset + Encoder.FRAME_SIZE]
            if not frame:
                break
            if len(frame) < Encoder.FRAME_SIZE:
                # To avoid "clatz" in the end
                frame += b'\x00' * (Encoder.FRAME_SIZE - len(frame))
            self.client.send_audio_packet(frame)
            when_to_wake += Encoder.FRAME_LENGTH / 1000
            to_sleep = when_to_wake - time.perf_counter()
            await asyncio.sleep(to_sleep)

    async def play_loop(self, sound: bytes):
        while True:
            await self.play(sound)

    async def cleanup(self):
        if self.deleted:
            return
        self.deleted = True
        logger.debug('Deleting Porcupine')
        for usersink in self.users.values():
            await usersink.close()


class DialogflowCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger('discord.gateway').setLevel(logging.ERROR)

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


class Registry:
    def __init__(self):
        self.skills = {}

    def skill(self, name):
        def decorator(func):
            self.skills[name] = func
            return func
        return decorator

    async def run_skill(self, bot, skill, parameters):
        if skill in self.skills:
            await self.skills[skill](bot, parameters)
        else:
            await bot.speak("такого я не умею")


registry = Registry()


@registry.skill('quest')
async def dfrotz_skill(bot, parameters):
    await bot.speak("стартую скиллуху")


@bot.event
async def on_ready():
    print('Logged in as {0} ({0.id})'.format(bot.user))
    print('------')
    _load_default()
    channel = bot.get_channel(653229977545998350)
    voice_client = await channel.connect()
    sink = DemultiplexerSink(voice_client, [
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
    await sink.play(sink.hello)

bot.add_cog(DialogflowCog(bot))
bot.run(os.getenv('TOKEN'))
