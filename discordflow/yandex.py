import json
import logging
import os

import aiohttp

from .utils import Audio, language

logger = logging.getLogger(__name__)


def get_lang():
    return dict(ru='ru-RU', en='en-US')[language.get()]


async def request_yandex(url: str, **kwargs):
    api_key = os.getenv('YANDEX_API_KEY')
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, **kwargs, headers=dict(Authorization=f'Api-Key {api_key}')) as response:
                body = await response.read()
                response.raise_for_status()
                return body
        except aiohttp.client_exceptions.ClientResponseError as exc:
            logger.exception(f"Unexpected exception while requesting Yandex.API: {exc} ({body})")
            raise


async def text_to_speech(text=None, ssml=None) -> Audio:
    logger.info(f"TTS: {text or ssml}")
    rate = 48000
    if text == ssml:
        raise ValueError("One and Only one of (text, ssml) must be set")
    elif text:
        kwargs = dict(text=text)
    elif ssml:
        kwargs = dict(ssml=ssml)
    data = await request_yandex(
        'https://tts.api.cloud.yandex.net/speech/v1/tts:synthesize',
        data=dict(format='lpcm', voice='alena', sampleRateHertz=rate, lang=get_lang(), **kwargs),
    )
    return Audio(data=data, channels=1, rate=rate, width=2)


async def speech_to_text(speech: Audio):
    response = await request_yandex(
        'https://stt.api.cloud.yandex.net/speech/v1/stt:recognize',
        params=dict(format='lpcm', sampleRateHertz=speech.rate, lang=get_lang()),  # TODO: add channels and width
        data=speech.to_mono().data,
    )
    result = json.loads(response)['result']
    logger.info(f"STT: {result}")
    return result
