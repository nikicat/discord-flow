import asyncio
import audioop
import io
import logging
import logging.handlers
import os
import importlib
import random
import re
import time
from contextlib import suppress
from copy import deepcopy
from itertools import count, groupby
from struct import Struct

import pkg_resources
importlib.reload(pkg_resources)  # HACK: https://github.com/googleapis/google-api-python-client/issues/476#issuecomment-371797043

from discord.ext import commands
from discord.opus import Decoder, Encoder, _load_default
from discord.reader import AudioSink
from pvporcupine import create
from serpapi.google_search_results import GoogleSearchResults

from .utils import sync_to_async, load_sound, BackgroundTask, Interrupted, SAMPLE_WIDTH
from .yandex import text_to_speech, speech_to_text
from .google import detect_intent

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PorcupineSink:
    def __init__(self, parent, keywords, user):
        self.parent = parent
        self.user = user
        self.pcp = create(keywords=keywords, sensitivities=[0.5] * len(keywords))
        self.audio_state = None
        self.input = b''
        self.question_input = b''
        self.unpacker = Struct(f'{self.pcp.frame_length}h')
        self.input_queue = asyncio.Queue()

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

    async def listen_utterance(self, timeout=2.3):
        logger.debug(f"Listening utterance with timeout {timeout}")
        self.background_player.start(self.play_loop(self.ticktock))
        sound = b''
        speech_started = False
        while True:
            try:
                actual_timeout = 0.4 if speech_started else timeout
                sound += await asyncio.wait_for(self.listen(), timeout=actual_timeout)
                # FIXME
                if len(sound) > 50000:
                    speech_started = True
            except asyncio.TimeoutError:
                logger.debug("Stop listening utterance due a timeout")
                await self.background_player.stop()
                return sound

    async def listen(self):
        return await self.input_queue.get()

    def write(self, data):
        self.input_queue.put_nowait(data.pcm)
        logger.log(5, f'write: {data.user}. qsize={self.input_queue.qsize()}')
        return

    async def play_interruptible(self, answer):
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

    async def process_utterance(self, utterance: bytes):
        text = await speech_to_text(utterance)
        response = await detect_intent(text, self.user)
        output_audio = response.output_audio
        response.output_audio = b''
        logger.log(5, f"Response: {response}")
        logger.debug(f"Query: {response.query_result.query_text}")
        logger.debug(f"Fulfillment: {response.query_result.fulfillment_text}")
        if output_audio:
            answer = load_sound(io.BytesIO(output_audio))
            mono_answer = audioop.tostereo(answer, 2, 1, 1)
            await self.play_interruptible(mono_answer)
        action = response.query_result.action
        parameters = response.query_result.parameters.fields
        query = parameters['query'].string_value if 'query' in parameters else response.query_result.query_text
        if output_audio and not response.query_result.all_required_params_present:
            user_answer = await self.listen_utterance()
            await self.process_utterance(user_answer)
        elif action == 'google.search':
            answer = await self.google_search(query)
            await self.play_interruptible(await text_to_speech(text=answer))
        elif action == 'format':
            answer = response.query_result.filfillment_text.format(query=query)
            await self.play_interruptible(answer)
        elif action == 'skill':
            skill = parameters['skill'].string_value
            params = {p: parameters[p].string_value for p in parameters}
            await registry.run_skill(self, skill, params)
        elif action is None and response.output_audio is None:
            logger.debug("Empty response")

    async def ask(self, question, timeout=4):
        request = await text_to_speech(text=question)
        await self.play_interruptible(request)
        reply = await self.listen_utterance(timeout=timeout)
        return await speech_to_text(reply)

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
        logging.basicConfig(level=logging.INFO)
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


def get_next_letter(city):
    return re.sub('[ьъ]', '', city)[-1]


@registry.skill('cities')
async def cities_skill(bot, parameters):
    cities = {
        first_letter: list(cities)
        for first_letter, cities in groupby(
            sorted([line.split(';')[-1][1:-1].lower() for line in open('city_.csv').read().split('\n')[1:]]),
            key=lambda city: city and city[0],
        )
    }
    orig_cities = deepcopy(cities)
    seen_cities = set()
    user_city = await bot.ask("Давай! Называй город!", timeout=5)
    bot_city = None
    while True:
        user_city = user_city.lower()
        if user_city == "повтори":
            user_city = await bot.ask(bot_city, timeout=5)
            continue
        user_letter = user_city[0]
        if user_city not in orig_cities[user_letter]:
            await bot.speak("Такого города нет, ты проиграл ха-ха-ха")
        elif bot_city and user_letter != (expected_user_letter := get_next_letter(bot_city)):
            await bot.speak(f"Неправильно, ты должен был назвать город на букву {expected_user_letter}. Ты проиграл, ха-ха-ха!")
        elif user_city in seen_cities:
            await bot.speak("Этот город уже называли, ты проиграл, ха-ха-ха!")
        else:
            seen_cities.add(user_city)
            cities[user_letter].remove(user_city)
            bot_letter = get_next_letter(user_city)
            bot_city = random.choice(cities[bot_letter])
            seen_cities.add(bot_city)
            cities[bot_letter].remove(bot_city)
            logger.debug(f"Bot city: {bot_city}")
            user_city = await bot.ask(bot_city, timeout=5)
            continue
        break


class DiscordHandler(logging.Handler):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.loop = asyncio.get_running_loop()

    async def send_message(self, message):
        await self.channel.send(f'```{message}```')

    def emit(self, record):
        asyncio.run_coroutine_threadsafe(self.send_message(self.format(record)), self.loop)


@bot.event
async def on_ready():
    print('Logged in as {0} ({0.id})'.format(bot.user))
    print('------')
    _load_default()
    handler = DiscordHandler(bot.get_channel(653229977545998349))
    logger.addHandler(handler)
    voice_channel = bot.get_channel(653229977545998350)
    voice_client = await voice_channel.connect()
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


def main():
    bot.add_cog(DialogflowCog(bot))
    bot.run(os.getenv('TOKEN'))


if __name__ == '__main__':
    main()
