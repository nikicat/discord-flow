import asyncio
import logging
import logging.handlers
import os
import importlib
import time
from contextlib import suppress, asynccontextmanager
from struct import Struct
from typing import List

import pkg_resources
import webrtcvad
importlib.reload(pkg_resources)  # HACK: https://github.com/googleapis/google-api-python-client/issues/476#issuecomment-371797043

from discord import File, SpeakingState, VoiceClient
from discord.ext import commands
from discord.opus import Decoder, Encoder, _load_default
from discord.reader import AudioSink
from pvporcupine import create
from serpapi.google_search_results import GoogleSearchResults

from .utils import sync_to_async, BackgroundTask, Interrupted, Audio, registry, background_task, EmptyUtterance
from .yandex import text_to_speech, speech_to_text
from .google import detect_intent

logger = logging.getLogger(__name__)


class PorcupineSink:
    def __init__(self, parent, keywords, user):
        self.parent = parent
        self.user = user
        self.pcp = create(keywords=keywords, sensitivities=[0.5] * len(keywords))
        self.unpacker = Struct(f'{self.pcp.frame_length}h')
        self.input_queue = asyncio.Queue()

        self.what = parent.what
        self.that = parent.that
        self.ticktock = parent.ticktock
        self.keywords = parent.keywords
        self.play = parent.play
        self.play_loop = parent.play_loop
        self.play_stream = parent.play_stream

        self.listen_task = BackgroundTask()
        self.listen_task.start(self.main_loop())
        self.vad = webrtcvad.Vad(3)

        logger.debug(f'Started Porcupine sink {self!r}')

    def detect_wuw(self, sound: Audio):
        result = self.pcp.process(self.unpacker.unpack(sound.data))
        if result >= 0:
            return self.keywords[result]

    async def main_loop(self):
        while True:
            try:
                await self.wait_for_wuw()
                utterance: Audio = await self.listen_utterance()
                await self.process_utterance(utterance)
            except EmptyUtterance:
                await self.speak("В следующий раз я тоже тебе не отвечу!")
            except Interrupted:
                pass
            except Exception as exc:
                logger.exception(f"Exception in main loop: {exc}")

    async def wait_for_wuw(self) -> str:
        """Listen for wake up word and return it"""
        sound = Audio(rate=self.pcp.sample_rate, channels=1, width=2)
        while True:
            sound += (await self.listen()).to_mono().to_rate(self.pcp.sample_rate)
            while len(sound) >= self.pcp.frame_length:
                to_process = sound[:self.pcp.frame_length]
                sound = sound[self.pcp.frame_length:]
                keyword = self.detect_wuw(to_process)
                if keyword:
                    logger.debug(f'Detected keyword "{keyword}"')
                    return keyword

    async def listen_utterance(self, timeout=2.3) -> Audio:
        logger.debug(f"Listening utterance with timeout {timeout}")
        async with background_task(self.play_loop(self.ticktock)):
            sound = Audio(channels=Decoder.CHANNELS, width=2, rate=Decoder.SAMPLING_RATE)
            speech_count = 0
            speech_threshold = 5
            while True:
                try:
                    actual_timeout = 0.4 if speech_count >= speech_threshold else timeout
                    packet = await asyncio.wait_for(self.listen(), timeout=actual_timeout)
                    sound += packet
                    is_speech = self.vad.is_speech(packet.to_mono().to_rate(16000).data, 16000)
                    logger.debug(f"Speech packet: is_speech={is_speech} rms={packet.rms}")
                    if is_speech:
                        speech_count += 1
                except asyncio.TimeoutError:
                    logger.info("Stop listening utterance")
                    if len(sound) == 0:
                        raise EmptyUtterance()
                    return sound

    async def listen(self) -> Audio:
        return await self.input_queue.get()

    def write(self, data):
        self.input_queue.put_nowait(Audio(data=data.pcm, channels=2, width=2, rate=48000))
        logger.log(5, f'write: {data.user}. qsize={self.input_queue.qsize()}')

    async def play_interruptible(self, answer: Audio):
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

    async def process_utterance(self, utterance: Audio):
        text = await speech_to_text(utterance)
        if not text:
            raise EmptyUtterance()
        intent = await detect_intent(text, self.user)
        logger.info(f"Response: {intent}")
        query = intent.parameters.get('query', text)
        if intent.text:
            await self.speak(intent.text)
            if not intent.all_required_params_present:
                user_answer: Audio = await self.listen_utterance()
                await self.process_utterance(user_answer)
        if intent.action == 'google.search':
            answer = await self.google_search(query)
            await self.speak(answer)
        elif intent.action == 'format':
            answer = intent.text.format(query=query)
            await self.speak(answer)
        elif intent.action == 'skill':
            skill = intent.parameters['skill']
            await registry.run_skill(skill, self, self.user, **intent.parameters)
        elif intent.action:
            logger.warning(f"Unknown action {intent.action}")
            await self.speak(f"Неизвестный экшен: {intent.action}")

    async def ask(self, question, timeout=4, tries=1):
        request = await text_to_speech(text=question)
        await self.play_interruptible(request)
        while True:
            try:
                speech = await self.listen_utterance(timeout=timeout)
                text = await speech_to_text(speech)
                if not text:
                    raise EmptyUtterance()
                return text
            except EmptyUtterance:
                tries -= 1
                if tries:
                    await self.speak("Ну что же ты молчишь?")
                else:
                    raise

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
        logger.debug(f"Speaking: {text}")
        audio = await text_to_speech(text=text)
        await self.play_interruptible(audio)

    def ddg_search(self, query):
        pass

    async def close(self):
        await self.listen_task.stop()
        self.pcp.delete()


async def aiter(iter_):
    for i in iter_:
        yield i


async def size_limit(audio_iter, size):
    buf = None
    async for packet in audio_iter:
        buf = buf + packet if buf else packet
        while len(buf) >= size:
            yield buf[:size]
            buf = buf[size:]
    if buf:
        yield buf


async def rate_limit(audio_iter):
    when_to_wake = time.perf_counter()
    async for packet in audio_iter:
        yield packet
        when_to_wake += packet.duration
        to_sleep = when_to_wake - time.perf_counter()
        await asyncio.sleep(to_sleep)
    if packet:
        yield packet


class DemultiplexerSink(AudioSink):
    def __init__(self, voice_client: VoiceClient, keywords: List[str]):
        self.client = voice_client
        self.keywords = keywords
        self.users = {}
        self.deleted = False
        self.is_speaking = False
        self.what = Audio.load('what2.wav')
        self.that = Audio.load('that2.wav')
        self.hello = Audio.load('hello.wav')
        self.ticktock = Audio.load('ticktock.wav')
        voice_client.listen(self)
        logger.debug('Started Demultiplexer sink')

    def write(self, data):
        if data.user not in self.users:
            self.users[data.user] = PorcupineSink(self, self.keywords, data.user)
        self.users[data.user].write(data)

    async def play_stream(self, stream):
        async with self.speaking():
            async for frame in rate_limit(size_limit(stream, Encoder.SAMPLES_PER_FRAME)):
                await self.send_packet(frame)

    async def play(self, sound: Audio):
        async with self.speaking():
            async for packet in rate_limit(size_limit(aiter([sound]), Encoder.SAMPLES_PER_FRAME)):
                if len(packet) < Encoder.SAMPLES_PER_FRAME:
                    # To avoid "clatz" in the end
                    packet += packet.silence(Encoder.SAMPLES_PER_FRAME - len(packet))
                await self.send_packet(packet)

    async def send_packet(self, packet: Audio):
        self.client.send_audio_packet(packet.to_stereo().to_rate(Encoder.SAMPLING_RATE).data)

    async def play_loop(self, sound: Audio):
        async with self.speaking():
            while True:
                await self.play(sound)

    @asynccontextmanager
    async def speaking(self):
        if self.is_speaking:
            yield
        else:
            try:
                await self.client.ws.speak(SpeakingState.active())
                self.is_speaking = True
                yield
            finally:
                self.is_speaking = False
                await self.client.ws.speak(SpeakingState.inactive())

    async def cleanup(self):
        if self.deleted:
            return
        self.deleted = True
        logger.debug('Deleting Porcupine')
        for usersink in self.users.values():
            await usersink.close()


class DialogflowCog(commands.Cog):
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

    @commands.command()
    async def youtube_dl(self, ctx, url: str):
        skill_ctx = voice_bot.users[ctx.author]
        await registry.run_skill('youtube-dl', skill_ctx, ctx.author, url)

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


@registry.skill('quest')
async def dfrotz_skill(bot, parameters):
    await bot.speak("стартую скиллуху")


class DiscordHandler(logging.Handler):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.loop = asyncio.get_running_loop()

    async def send_message(self, speech: Audio, message: str):
        await self.channel.send(f'```{message}```', file=speech and File(fp=speech.to_wav(), filename='speech.wav'))

    def emit(self, record: logging.LogRecord):
        asyncio.run_coroutine_threadsafe(self.send_message(getattr(record, 'speech', None), self.format(record)), self.loop)


bot = commands.Bot(command_prefix=commands.when_mentioned_or("!"), description='Relatively simple music bot example')
voice_bot = None


@bot.event
async def on_ready():
    logger.info(f"Logged in as {bot.user}")
    _load_default()
    handler = DiscordHandler(bot.get_channel(653229977545998349))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s: %(message)s')
    handler.setFormatter(formatter)
    for logger_name in os.getenv('LOGGERS_DISCORD').split(','):
        logging.getLogger(logger_name).addHandler(handler)
    voice_channel = bot.get_channel(653229977545998350)
    voice_client = await voice_channel.connect()
    global voice_bot
    voice_bot = DemultiplexerSink(voice_client, [
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
    await voice_bot.play(voice_bot.hello)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s[%(name)-20s] %(message)s')
    logging.getLogger('discord.gateway').setLevel(logging.WARNING)
    for level, name in logging._levelToName.items():
        for logger_name in os.getenv(f'LOGGERS_{name}', '').split(','):
            if logger_name:
                logging.getLogger(logger_name).setLevel(level)


def main():
    setup_logging()
    from . import cities, youtubedl  # noqa to load skills
    bot.add_cog(DialogflowCog())
    bot.run(os.getenv('TOKEN'))
