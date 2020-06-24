import json
import logging
import os

import aiohttp

from .utils import Audio

logger = logging.getLogger(__name__)


async def request_yandex(url: str, **kwargs):
    api_key = os.getenv('YANDEX_API_KEY')
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.post(url, **kwargs, headers=dict(Authorization=f'Api-Key {api_key}')) as response:
            return await response.read()


async def text_to_speech(**data) -> Audio:
    rate = 48000
    data = await request_yandex(
        'https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize',
        data=dict(format='lpcm', sampleRateHertz=rate, **data),
    )
    return Audio(data=data, channels=1, rate=rate, width=2)


async def speech_to_text(speech: Audio):
    response = await request_yandex(
        'https://stt.api.cloud.yandex.net/speech/v1/stt:recognize',
        params=dict(format='lpcm', sampleRateHertz=speech.rate),  # TODO: add channels and width
        data=speech.to_mono().data,
    )
    result = json.loads(response)['result']
    logger.info(f"STT: {result}", extra=dict(speech=speech))
    return result
