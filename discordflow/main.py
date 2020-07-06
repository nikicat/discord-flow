import asyncio
import logging
import logging.handlers
import os
from contextlib import suppress, asynccontextmanager, contextmanager
from itertools import accumulate
from struct import Struct
from typing import List, AsyncGenerator, Dict

import discord.utils
from discord import File, SpeakingState, VoiceClient, TextChannel, Guild, User
from discord.ext import commands
from discord.opus import Decoder, Encoder, _load_default
from discord.reader import AudioSink, AudioReader
from pvporcupine import create

from . import yandex, google
from .google import detect_intent, set_contexts
from .utils import (
    BackgroundTask, Interrupted, Audio, registry, background_task, EmptyUtterance, rate_limit, size_limit, aiter, language,
    TooLongUtterance,
)
from .skills import *  # noqa to load skills

logger = logging.getLogger(__name__)


class ListenMore(Exception):
    pass


class EmptyIntent(Exception):
    pass


async def speech_to_text(audio: Audio):
    if language.get() == 'ru':
        return await yandex.speech_to_text(audio)
    else:
        return await google.speech_to_text(audio)


async def text_to_speech(text: str):
    if language.get() == 'ru':
        return await yandex.text_to_speech(text)
    else:
        return await google.text_to_speech(text)


async def speech_stream_to_text(speech_stream: AsyncGenerator[Audio, None]):
    if language.get() == 'ru':
        async for resp in yandex.speech_stream_to_text(speech_stream):
            pass
        return resp[0].text
    else:
        # TODO: implement Google streaming STT
        return await google.speech_to_text(accumulate(packet async for packet in speech_stream))


class VoiceUserContext:
    def __init__(self, parent, keywords, user):
        self.parent = parent
        self.user = user
        self.pcp = create(keywords=keywords, sensitivities=[0.5] * len(keywords))
        self.unpacker = Struct(f'{self.pcp.frame_length}h')
        self.welcome_response = None
        self.wuw_input = asyncio.Queue()
        self.input_queue = self.wuw_input

        self.what = parent.what
        self.that = parent.that
        self.ticktock = parent.ticktock
        self.keywords = parent.keywords
        self.play = parent.play
        self.play_loop = parent.play_loop
        self.play_stream = parent.play_stream
        self.play_interruptible = parent.play_interruptible
        self.say = parent.say
        self.listen_task = BackgroundTask()
        self.listen_task.start(self.listen_loop())
        logger.info(f"Started {self!r}")

    def __repr__(self):
        return f'{type(self).__name__}<{self.parent}:{self.user}>'

    def detect_wuw(self, sound: Audio):
        result = self.pcp.process(self.unpacker.unpack(sound.data))
        if result >= 0:
            return self.keywords[result]

    async def process_wakeup(self):
        try:
            with set_contexts():
                while True:
                    try:
                        text = await self.listen_text()
                    except TooLongUtterance:
                        await self.say("Скажи покороче")
                        continue
                    try:
                        await self.process_intent(text)
                    except ListenMore:
                        logger.debug("Intent wants more info, listening again")
                    else:
                        break
        except EmptyUtterance:
            await self.say("В следующий раз я тоже тебе не отвечу!")
        except Interrupted:
            logger.info(f"{self!r} interrupted")

    async def wait_for_wuw(self) -> str:
        """Listen for wake up word and return it"""
        sound = Audio(rate=self.pcp.sample_rate, channels=1, width=2)
        while True:
            sound += (await self.wuw_input.get()).to_mono().to_rate(self.pcp.sample_rate)
            while len(sound) >= self.pcp.frame_length:
                to_process = sound[:self.pcp.frame_length]
                sound = sound[self.pcp.frame_length:]
                keyword = self.detect_wuw(to_process)
                if keyword:
                    logger.debug(f'Detected keyword "{keyword}"')
                    return keyword

    async def listen_loop(self):
        while True:
            try:
                await self.wait_for_wuw()
                async with self.parent.attention:
                    await self.process_wakeup()
            except Exception as exc:
                logger.exception(f"Unexpected exception in {self!r}.listen_loop: {exc}")

    async def listen_audio(self, timeout=2.3) -> Audio:
        speech = Audio(channels=Decoder.CHANNELS, width=2, rate=Decoder.SAMPLING_RATE)
        async for packet in self.listen_audio_stream(timeout=timeout):
            speech += packet
        return speech

    @contextmanager
    def listening_utterance(self):
        try:
            self.input_queue = asyncio.Queue()
            yield self.input_queue
        finally:
            self.input_queue = self.wuw_input

    async def listen_audio_stream(self, timeout=2.3) -> AsyncGenerator[Audio, None]:
        logger.info(f"{self!r} start listening utterance with timeout {timeout}")
        async with background_task(self.play_loop(self.ticktock)):
            sound = Audio(channels=Decoder.CHANNELS, width=2, rate=Decoder.SAMPLING_RATE)
            speech_count = 0
            speech_threshold = 10
            with self.listening_utterance() as queue:
                while True:
                    try:
                        actual_timeout = 0.4 if speech_count >= speech_threshold else timeout
                        packet = await asyncio.wait_for(queue.get(), timeout=actual_timeout)
                        yield packet
                        sound += packet
                        logger.debug(f"{self!r} speech packet: rms={packet.rms}")
                        if packet.rms > 50:
                            speech_count += 1
                    except asyncio.TimeoutError:
                        if not sound:
                            logger.info(f"{self!r} speech was empty")
                            raise EmptyUtterance
                        logger.info(f"{self!r} listened speech: {sound.duration}s", extra=dict(speech=sound))
                        break

    async def listen_text(self, timeout=4, tries=1):
        while True:
            try:
                speech_stream = self.listen_audio_stream(timeout=timeout)
                text = await speech_stream_to_text(speech_stream)
                if not text:
                    raise EmptyUtterance
                return text
            except EmptyUtterance:
                tries -= 1
                if tries:
                    await self.say("Ну то же ты молчишь?")
                else:
                    raise

    async def feed(self, audio: Audio):
        self.input_queue.put_nowait(audio)
        logger.debug(f'feed: {self!r}. qsize={self.input_queue.qsize()}')

    async def process_intent(self, text: str = None, speech: Audio = None, event: str = None, params: dict = None):
        intent = await detect_intent(self.user, text=text, speech=speech, event=event, params=params)
        query = intent.parameters.get('query', text)
        if not intent.all_required_params_present:
            await self.say(intent.text)
            raise ListenMore
        elif intent.action == 'format':
            answer = intent.text.format(query=query)
            await self.say(answer)
        elif intent.action.startswith('skill.'):
            skill = intent.action.split('skill.')[-1]
            await registry.run_skill(skill, self, **intent.parameters)
        elif intent.action == 'set-language':
            language.set(intent.parameters['lang'])
        elif intent.action == 'fallback':
            await registry.run_skill('parlai', self, initial=intent.query_text)
        elif intent.action == 'responses':
            responses = [
                f"{user.nick or user.name} говорит, что {ctx.welcome_response}" for user, ctx in self.parent.users.items()
                if ctx.welcome_response
            ]
            if responses:
                text = ". ".join(responses)
            else:
                text = "Пока новостей нет"
            await self.say(text)
        elif intent.action:
            logger.warning(f"{self!r} unknown action {intent.action}")
            await self.say(f"неизвестный экшен: {intent.action}")
        elif intent.text:
            await self.say(intent.text)
        else:
            raise EmptyIntent

    async def ask(self, question, timeout=4, tries=1):
        request = await text_to_speech(text=question)
        await self.play(request)
        return await self.listen_text(timeout, tries)

    async def ask_and_detect_intent(self, question, timeout=4, tries=3):
        await self.say(question)
        while (tries := tries - 1) >= 0:
            try:
                speech = await self.listen_audio(timeout=timeout)
            except EmptyUtterance:
                await self.say("Ну что же ты молчишь?")
            else:
                intent = await detect_intent(self.user, speech=speech)
                if intent.action == 'repeat':
                    await self.say(question)
                    tries += 1
                elif intent.action == 'fallback':
                    await self.say(intent.text)
                else:
                    return intent
        else:
            raise EmptyUtterance

    async def ask_yes_no(self, question: str):
        with set_contexts('yes-no'):
            intent = await self.ask_and_detect_intent(question)
            return intent.action == 'yes'

    async def close(self):
        self.pcp.delete()

    async def on_welcome(self):
        logger.info(f"Welcome {self.user}")
        name = self.user.nick or self.user.name
        if not self.welcome_response:
            last_welcomed_user = self.parent.last_welcomed_user
            if last_welcomed_user:
                last_name = last_welcomed_user.nick or last_welcomed_user.name
                last_response = self.parent.user[last_welcomed_user].welcome_response
                text = f"Привет, {name}! {last_name} сказал, что {last_response}. А у тебя какие новости?"
            else:
                text = f"Привет, {name}! Что нового?"
            self.welcome_response = await self.ask(text)
        self.parent.last_welcomed_user = self.user


class TextUserContext:
    def __init__(self, channel: TextChannel, user):
        self.channel = channel
        self.user = user

    async def say(self, text: str):
        await self.channel.send(text)

    async def ask(self, text: str) -> str:
        await self.say(text)
        await self.client.wait_for('message', lambda m: m.channel == self.channel and m.author == self.user)


class ChannelVoiceContext(AudioSink):
    def __init__(self, voice_client: VoiceClient, keywords: List[str]):
        self.client = voice_client
        self.keywords = keywords
        self.users: Dict[User, VoiceUserContext] = {}
        self.deleted = False
        self.is_speaking = False
        self.what = Audio.load('what2.wav')
        self.that = Audio.load('that2.wav')
        self.hello = Audio.load('hello.wav')
        self.ticktock = Audio.load('ticktock.wav')
        self.wakeups = asyncio.Queue()
        self.reader = AudioReader(self, voice_client)
        self.demux_task = BackgroundTask()
        self.attention = asyncio.Lock()
        self.speak_lock = asyncio.Lock()
        self.last_welcomed_user = None

    def __str__(self):
        return f'guild={self.client.guild}'

    def __repr__(self):
        return f'<{type(self).__name__}>[{self}]'

    async def start(self):
        self.demux_task.start(self.demux_loop())
        logger.debug(f"Started {self!r}")
        await self.play(self.hello)

    async def cleanup(self):
        # TODO: move to __aenter__/__aexit__
        if self.deleted:
            return
        self.deleted = True
        logger.debug(f"Deleting {self!r}")
        for usersink in self.users.values():
            await usersink.close()

    async def demux_loop(self):
        async for voice_data in self.reader.listen_voice():
            try:
                sink = self.get_user_sink(voice_data.user)
                audio = Audio(data=voice_data.pcm, channels=2, width=2, rate=48000)
                await sink.feed(audio)
            except Exception as exc:
                logger.exception(f"Unexpected exception in {self!r}.demux_loop: {exc}")

    def get_user_sink(self, user):
        if user not in self.users:
            self.users[user] = VoiceUserContext(self, self.keywords, user)
        return self.users[user]

    async def play_stream(self, stream):
        return await self.run_interruptible(self.play_stream_impl(stream))

    async def play_stream_impl(self, stream):
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

    async def run_interruptible(self, coro):
        logger.debug("Playing interruptible")
        # FIXME: listen for users those entered chat after start
        waiters = {sink.wait_for_wuw(): user for user, sink in self.users.items()}
        done, pending = await asyncio.wait(list(waiters) + [coro], return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        done_coro = done.pop().get_coro()
        if done_coro in waiters:
            logger.debug(f"Interrupted by {waiters[done_coro]}")
            raise Interrupted

    async def play_interruptible(self, audio: Audio):
        return await self.run_interruptible(self.play(audio))

    async def say(self, text):
        logger.debug(f"Speaking: {text}")
        audio = await text_to_speech(text=text)
        await self.play_interruptible(audio)

    @asynccontextmanager
    async def speaking(self):
        if self.is_speaking:
            yield
        else:
            try:
                async with self.speak_lock:
                    await self.client.ws.speak(SpeakingState.active())
                    self.is_speaking = True
                    yield
            finally:
                self.is_speaking = False
                await self.client.ws.speak(SpeakingState.inactive())

    async def on_welcome(self, user):
        await self.get_user_sink(user).on_welcome()


class DialogflowCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.voice_bots: Dict[Guild, ChannelVoiceContext] = {}
        self.text_contexts = {}

    def get_text_context(self, ctx):
        key = (ctx.channel, ctx.author)
        if key not in self.text_contexts:
            self.text_contexts[key] = TextUserContext(ctx.channel, ctx.author)
        return self.text_contexts[key]

    @commands.command()
    async def voice(self, ctx):
        """Joins a voice channel"""
        if ctx.voice_client.channel != ctx.author.voice.channel:
            return await ctx.voice_client.move_to(ctx.author.voice.channel)

    @commands.command()
    async def youtube_dl(self, ctx, url: str):
        # TODO: how to get voice channel from text channel???
        skill_ctx = ctx.bot.voice_bots[ctx.guild].users[ctx.author]
        await registry.run_skill('youtube-dl', skill_ctx, url)

    @commands.command()
    async def parlai(self, ctx, initial: str = None):
        skill_ctx = self.get_text_context(ctx)
        await registry.run_skill('parlai', skill_ctx, initial)

    @commands.command()
    async def search(self, ctx, query):
        skill_ctx = self.get_text_context(ctx)
        await registry.run_skill('duckduckgo', skill_ctx, query)

    @commands.command()
    async def responses(self, ctx):
        bot = self.voice_bots[ctx.guild]
        responses = [f" - {user}: {ctx.welcome_response} " for user, ctx in bot.users.items() if ctx.welcome_response]
        await ctx.send("\n".join(responses))

    @commands.Cog.listener()
    async def on_ready(self):
        logger.info(f"Logged in as {self.bot.user}")
        _load_default()
        channels = {channel.name: channel for channel in self.bot.get_all_channels()}
        handler = DiscordHandler(channels['bot-logs'])
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)s: %(message)s')
        handler.setFormatter(formatter)
        for logger_name in os.getenv('LOGGERS_DISCORD').split(','):
            logging.getLogger(logger_name).addHandler(handler)
        logger.debug(f"Guilds: {self.bot.guilds}")
        for guild in self.bot.guilds:
            logger.debug(f"{guild.name} channels: {guild.channels}")
            logger.debug(f"{guild.name} voice channels: {guild.voice_channels}")
            # TODO: save and use last voice channel
            voice_channel = discord.utils.get(guild.voice_channels)
            voice_client = await voice_channel.connect()
            voice_bot = ChannelVoiceContext(voice_client, [
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
            await voice_bot.start()
            self.voice_bots[guild] = voice_bot

    @commands.Cog.listener()
    async def on_guild_join(self, guild):
        logger.debug(f"{self!r} joined {guild}")

    @commands.Cog.listener()
    async def on_group_join(self, channel, user):
        logger.debug(f"{self!r} observed that {user} joined {channel}")

    @commands.Cog.listener()
    async def on_voice_state_update(self, user, old_state, new_state):
        if user != self.bot.user and old_state.channel is None and new_state.channel is not None:
            await asyncio.sleep(0.4)  # Wait while user voice connection is established
            await self.voice_bots[new_state.channel.guild].on_welcome(user)


class DiscordHandler(logging.Handler):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.loop = asyncio.get_running_loop()

    async def send_message(self, speech: Audio, message: str):
        await self.channel.send(f'```{message}```', file=speech and File(fp=speech.to_wav(), filename='speech.wav'))

    def emit(self, record: logging.LogRecord):
        asyncio.run_coroutine_threadsafe(self.send_message(getattr(record, 'speech', None), self.format(record)), self.loop)


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)-5s[%(name)-30s:%(lineno)d] %(message)s')
    logging.getLogger('discord.gateway').setLevel(logging.ERROR)  # Too noisy
    for level, name in logging._levelToName.items():
        for logger_name in os.getenv(f'LOGGERS_{name}', '').split(','):
            if logger_name:
                logging.getLogger(logger_name).setLevel(level)


def main():
    setup_logging()
    logger.info(f"Loaded skills: {list(registry.skills)}")
    asyncio.run(amain())


async def amain():
    bot = commands.bot.Bot(command_prefix=commands.when_mentioned_or('!'), description="Bot that can talk with group of people")
    bot.add_cog(DialogflowCog(bot))
    try:
        await bot.start(os.getenv('TOKEN'))
    finally:
        for task in asyncio.all_tasks():
            task.print_stack()
        await bot.close()
