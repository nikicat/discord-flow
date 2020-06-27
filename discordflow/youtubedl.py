import asyncio
import logging

import youtube_dl
from discord.opus import Encoder

from .utils import registry, sync_to_async, Audio


logger = logging.getLogger(__name__)


async def play_ffmpeg(url):
    proc = await asyncio.create_subprocess_exec(
        'ffmpeg', '-i', url, *('-f s16le -ar 48000 -ac 2 -vn -loglevel warning pipe:1'.split()),
        stdout=asyncio.subprocess.PIPE,
        limit=2 ** 20,
    )
    try:
        while True:
            data = await proc.stdout.read(Encoder.SAMPLES_PER_FRAME * 2 * 2)
            yield Audio(data=data, width=2, channels=2, rate=48000)
    finally:
        proc.terminate()
        await proc.stdout.read()
        await proc.wait()


@registry.skill('youtube-dl')
async def youtubedl(bot, user_state, url):
    ytdl = youtube_dl.YoutubeDL(dict(format='bestaudio/best'))
    data = await sync_to_async(ytdl.extract_info, url, download=False)
    if 'entries' in data:
        # take first item from a playlist
        data = data['entries'][0]
    await bot.play_stream(play_ffmpeg(data['url']))
