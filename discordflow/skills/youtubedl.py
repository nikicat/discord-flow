import asyncio
import logging

import aiohttp
import youtube_dl
from discord.opus import Encoder

from ..utils import registry, sync_to_async, Audio, background_task


logger = logging.getLogger(__name__)


async def fetch(url):
    logger.debug(f"Feetching {url}")
    async with aiohttp.ClientSession() as sess:
        read = 0
        while True:
            try:
                logger.debug(f"Requesting range [{read}:]")
                async with sess.get(url, headers=dict(Range=f'bytes={read}-')) as resp:
                    async for chunk in resp.content.iter_any():
                        logger.debug(f"Received chunk [{read}:{read+len(chunk)}]")
                        yield chunk
                        read += len(chunk)
                    else:
                        break
            except aiohttp.ClientError as exc:
                logger.debug(f"Error {exc!r} occurred while reading {url}, retrying")


async def feed_input(source, target):
    async for chunk in source:
        target.write(chunk)
        await target.drain()
    target.write_eof()
    await target.drain()


async def play_ffmpeg(url):
    proc = await asyncio.create_subprocess_exec(
        'ffmpeg', *('-i - -f s16le -ar 48000 -ac 2 -vn -loglevel warning pipe:1'.split()),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        limit=2 ** 20,
    )
    try:
        async with background_task(feed_input(fetch(url), proc.stdin)):
            while True:
                data = await proc.stdout.read(Encoder.SAMPLES_PER_FRAME * 2 * 2)
                yield Audio(data=data, width=2, channels=2, rate=48000)
    finally:
        proc.terminate()
        await proc.stdout.read()
        await proc.wait()


@registry.skill()
async def youtubedl(ctx, state, url):
    with youtube_dl.YoutubeDL(dict(format='bestaudio/best', verbose=True)) as ytdl:
        data = await sync_to_async(ytdl.extract_info, url, download=False)
    if 'entries' in data:
        # take first item from a playlist
        data = data['entries'][0]
    logger.info(f"Playing {data['title']}[{data['webpage_url']}]")
    await ctx.play_stream(play_ffmpeg(data['url']))


@registry.skill()
async def youtubedl_ytsearch(ctx, state, query, prefix=''):
    return await youtubedl(ctx, state, f'ytsearch:{prefix} {query}')
