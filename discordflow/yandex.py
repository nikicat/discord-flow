import json
import logging
import os
import audioop

import aiohttp

logger = logging.getLogger(__name__)


async def request_yandex(url: str, **kwargs):
    api_key = os.getenv('YANDEX_API_KEY')
    async with aiohttp.ClientSession(raise_for_status=True) as session:
        async with session.post(url, **kwargs, headers=dict(Authorization=f'Api-Key {api_key}')) as response:
            return await response.read()


async def text_to_speech(**data):
    data = await request_yandex(
        'https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize',
        data=dict(format='lpcm', sampleRateHertz=48000, **data),
    )
    return audioop.tostereo(data, 2, 1, 1)


async def speech_to_text(speech: bytes):
    response = await request_yandex(
        'https://stt.api.cloud.yandex.net/speech/v1/stt:recognize', params=dict(format='lpcm', sampleRateHertz=48000),
        data=speech,
    )
    result = json.loads(response)['result']
    logger.info(f"STT: {result}")
    return result
