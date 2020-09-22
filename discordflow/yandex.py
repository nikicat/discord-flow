import asyncio
import json
import logging
import os
from contextlib import suppress

import aiohttp
from grpclib.client import Channel
from grpclib.exceptions import ProtocolError
from yandex.cloud.ai.stt.v2.stt_service_pb2 import RecognitionConfig, RecognitionSpec, StreamingRecognitionRequest
from yandex.cloud.ai.stt.v2.stt_service_grpc import SttServiceStub

from .utils import Audio, language, TooLongUtterance, background_task, EmptyUtterance, cancellable_stream

logger = logging.getLogger(__name__)


def get_lang():
    return dict(ru='ru-RU', en='en-US')[language.get()]


def get_authorization_header():
    api_key = os.getenv('YANDEX_API_KEY')
    return f'Api-Key {api_key}'


async def request_yandex(url: str, **kwargs):
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(url, **kwargs, headers=dict(Authorization=get_authorization_header())) as response:
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
    if speech.duration > 10:
        raise TooLongUtterance
    response = await request_yandex(
        'https://stt.api.cloud.yandex.net/speech/v1/stt:recognize',
        params=dict(format='lpcm', sampleRateHertz=speech.rate, lang=get_lang()),  # TODO: add channels and width
        data=speech.to_mono().data,
    )
    result = json.loads(response)['result']
    logger.info(f"STT: {result}")
    return result


async def read_from_stream(stream):
    async for reply in stream:
        for chunk in reply.chunks:
            log = logger.info if chunk.final else logger.debug
            log(f"STT: {'|'.join(alt.text for alt in chunk.alternatives)}. Final={chunk.final}")
            yield chunk.alternatives
            if chunk.final:
                return
    else:
        raise EmptyUtterance


async def write_to_stream(stream, speech_stream, sent):
    try:
        async for speech in speech_stream:
            speech = speech.to_mono()
            if not sent.is_set():
                specification = RecognitionSpec(
                    language_code=get_lang(),
                    profanity_filter=False,
                    model='general',
                    partial_results=True,
                    audio_encoding='LINEAR16_PCM',
                    sample_rate_hertz=speech.rate,
                )
                streaming_config = RecognitionConfig(specification=specification)
                await stream.send_message(StreamingRecognitionRequest(config=streaming_config))
                sent.set()
            await stream.send_message(StreamingRecognitionRequest(audio_content=speech.data))
    finally:
        with suppress(ProtocolError):
            await stream.end()


@cancellable_stream
async def speech_stream_to_text(speech_stream):
    channel = Channel('stt.api.cloud.yandex.net', 443, ssl=True)
    try:
        stub = SttServiceStub(channel)
        sent = asyncio.Event()
        async with stub.StreamingRecognize.open(metadata=[('authorization', get_authorization_header())]) as stream:
            async with background_task(write_to_stream(stream, speech_stream, sent)):
                await sent.wait()
                async for text in read_from_stream(stream):
                    yield text
    finally:
        channel.close()
