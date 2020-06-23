import asyncio
import wave
from contextlib import suppress
from functools import partial

from discord.opus import Decoder

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


def load_sound(filename):
    return wave.open(filename, 'rb').readframes(100000)
