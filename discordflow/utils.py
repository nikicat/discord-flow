import asyncio
import audioop
import io
import wave
from contextlib import suppress
from dataclasses import dataclass
from functools import partial

import simpleaudio


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


@dataclass
class Audio:
    channels: int
    width: int
    rate: int
    data: bytes = b''

    def __str__(self):
        return f'channels={self.channels} width={self.width}B rate={self.rate}Hz frames={len(self)}'

    def __repr__(self):
        return f'{type(self)}[{self}]'

    def __add__(self, other):
        if other.channels != self.channels or other.width != self.width or other.rate != self.rate:
            raise ValueError("Could not add incompatible Audio")
        return Audio(channels=self.channels, width=self.width, rate=self.rate, data=self.data + other.data)

    def __getitem__(self, slice):
        """Slice frames"""
        start = slice.start and slice.start * self.framewidth
        stop = slice.stop and slice.stop * self.framewidth
        step = slice.step and slice.step * self.framewidth
        return Audio(channels=self.channels, width=self.width, rate=self.rate, data=self.data[start:stop:step])

    def __len__(self):
        """Audio length in frames"""
        return len(self.data) // self.framewidth

    @property
    def duration(self):
        """Duration in seconds"""
        return len(self) / self.rate

    @property
    def framewidth(self):
        return self.channels * self.width

    def clear(self):
        self.data = b''

    def to_mono(self):
        if self.channels == 1:
            return self
        elif self.channels == 2:
            return Audio(channels=1, width=self.width, rate=self.rate, data=audioop.tomono(self.data, self.width, 0.5, 0.5))
        else:
            raise ValueError(f"Can't convert audio with channels={self.channels}")

    def to_stereo(self):
        if self.channels == 2:
            return self
        elif self.channels == 1:
            return Audio(channels=2, width=self.width, rate=self.rate, data=audioop.tostereo(self.data, self.width, 0.5, 0.5))
        else:
            raise ValueError(f"Can't convert audio with channels={self.channels}")

    def to_rate(self, rate):
        converted, _ = audioop.ratecv(self.data, self.width, self.channels, self.rate, rate, None)
        return Audio(channels=self.channels, width=self.width, rate=rate, data=converted)

    @classmethod
    def load(cls, fp: str) -> 'Audio':
        with wave.open(fp, 'rb') as f:
            return Audio(
                data=f.readframes(100000),
                channels=f.getnchannels(),
                width=f.getsampwidth(),
                rate=f.getframerate(),
            )

    @classmethod
    def from_wav(cls, wav: bytes):
        return cls.load(io.BytesIO(wav))

    def to_wav(self):
        wav = io.BytesIO()
        with wave.open(wav, 'wb') as f:
            f.setnchannels(self.channels)
            f.setsampwidth(self.width)
            f.setframerate(self.rate)
            f.writeframes(self.data)
        wav.seek(0)
        return wav

    def silence(self, frames: int) -> 'Audio':
        return Audio(channels=self.channels, width=self.width, rate=self.rate, data=b'\x00' * (self.channels * self.width))

    def play(self):
        """Useful for debugging"""
        play = simpleaudio.play_buffer(self.data, num_channels=self.channels, bytes_per_sample=self.width, sample_rate=self.rate)
        try:
            play.wait_done()
        except KeyboardInterrupt:
            play.stop()
