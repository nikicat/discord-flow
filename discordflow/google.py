import asyncio
import html
import importlib
import logging
import os
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List

import aiohttp
import pkg_resources
importlib.reload(pkg_resources)  # HACK: https://github.com/googleapis/google-api-python-client/issues/476#issuecomment-371797043
from discord import Member
from google.auth.transport.aio.aiohttp import Request
from google.oauth2.aio.service_account import Credentials

from google.cloud.dialogflow.v2.audio_config_pb2 import InputAudioConfig, AudioEncoding
from google.cloud.dialogflow.v2.context_pb2 import Context
from google.cloud.dialogflow.v2.session_grpc import SessionsStub
from google.cloud.dialogflow.v2.session_pb2 import (
    QueryInput, TextInput, EventInput, QueryParameters, DetectIntentResponse, DetectIntentRequest,
)

from google.cloud.speech.v1.cloud_speech_pb2 import (
    StreamingRecognizeRequest, RecognitionConfig, StreamingRecognitionConfig, RecognizeRequest, RecognizeResponse,
    RecognitionAudio,
)
from google.cloud.speech.v1.cloud_speech_grpc import SpeechStub

from google.cloud.texttospeech.v1.cloud_tts_pb2 import (
    VoiceSelectionParams, SsmlVoiceGender, SynthesisInput, AudioConfig, SynthesizeSpeechRequest, SynthesizeSpeechResponse
)
from google.cloud.texttospeech.v1.cloud_tts_grpc import TextToSpeechStub

from google.cloud.translate.v3.translation_service_pb2 import TranslateTextRequest, TranslateTextResponse
from google.cloud.translate.v3.translation_service_grpc import TranslationServiceStub

from google.protobuf.struct_pb2 import Struct
from grpclib.client import Channel
from grpclib.exceptions import ProtocolError

from .utils import Audio, language, EmptyUtterance, background_task

logger = logging.getLogger(__name__)
LINEAR16 = 1  # FIXME


def get_lang():
    return dict(ru='ru-RU', en='en-IN')[language.get()]


async def text_to_speech(text) -> Audio:
    with create_channel('texttospeech.googleapis.com') as channel:
        client = TextToSpeechStub(channel)
        voice = VoiceSelectionParams(
            language_code=get_lang(),
            ssml_gender=SsmlVoiceGender.FEMALE,
            name='en-IN-Wavenet-B',
        )
        synthesis_input = SynthesisInput(text=text)
        rate = 48000
        audio_config = AudioConfig(
            audio_encoding=LINEAR16,
            sample_rate_hertz=rate,
        )
        request = SynthesizeSpeechRequest(input=synthesis_input, voice=voice, audio_config=audio_config)
        response: SynthesizeSpeechResponse = await client.SynthesizeSpeech(request, metadata=await get_metadata())
        result = Audio.from_wav(response.audio_content)
        logger.debug(f"TTS: {text} -> {result}")
        return result


async def speech_to_text(audio: Audio):
    with create_channel('speech.googleapis.com') as channel:
        client = SpeechStub(channel)
        request = RecognizeRequest(
            config=RecognitionConfig(
                language_code=get_lang(),
                sample_rate_hertz=audio.rate,
                encoding=LINEAR16,  # TODO: replace with AUDIO_ENCODING_OGG_OPUS
            ),
            audio=RecognitionAudio(content=audio.to_mono().data),
        )
        response: RecognizeResponse = await client.Recognize(request, metadata=await get_metadata())
        if not response.results:
            raise EmptyUtterance
        transcript = response.results[0].alternatives[0].transcript
        logger.info(f"STT: {response} {transcript}")
        return transcript


@dataclass
class Intent:
    query_text: str
    text: str
    parameters: Dict[str, str]
    action: str
    name: str
    all_required_params_present: bool
    output_contexts: List[str]
    input_contexts: List[str]
    response: DetectIntentResponse = None


LANGUAGE = 'ru'


def make_parameters(params: dict):
    parameters = Struct()
    parameters.update(params)
    return parameters


contexts_var = ContextVar('contexts', default=[])


@contextmanager
def set_contexts(*contexts):
    token = contexts_var.set(contexts)
    try:
        yield
    finally:
        contexts_var.reset(token)


async def detect_intent(user: Member, text: str = None, speech: Audio = None, event: str = None, params: dict = None) -> Intent:
    with create_channel('dialogflow.googleapis.com') as channel:
        dialogflow_project_id = os.getenv('DIALOGFLOW_PROJECT_ID')
        client = SessionsStub(channel)
        session = f'projects/{dialogflow_project_id}/agent/sessions/{user}'
        logger.debug(f"Session: {session}")
        kwargs = {}
        if text:
            query_input = QueryInput(text=TextInput(text=text, language_code=get_lang()))
        elif speech:
            query_input = QueryInput(audio_config=InputAudioConfig(
                audio_encoding=AudioEncoding.Value('AUDIO_ENCODING_LINEAR_16'),
                sample_rate_hertz=speech.rate,
                language_code=get_lang(),
                enable_word_info=True,
            ))
            kwargs = dict(input_audio=speech.to_mono().data)
        elif event:
            query_input = QueryInput(event=EventInput(name=event, parameters=make_parameters(params), language_code=get_lang()))
        else:
            raise ValueError("One of `text`, `speech` or `event` should be set")

        query_params = QueryParameters(
            contexts=[
                Context(name=f'{session}/contexts/{context}', lifespan_count=1)
                for context in contexts_var.get()
            ],
            reset_contexts=True,
        )

        response: DetectIntentResponse = await client.DetectIntent(
            DetectIntentRequest(
                session=session,
                query_input=query_input,
                query_params=query_params,
                **kwargs,
            ),
            metadata=await get_metadata(),
        )
        intent = Intent(
            text=response.query_result.fulfillment_text,
            parameters={field_name: field.string_value for field_name, field in response.query_result.parameters.fields.items()},
            action=response.query_result.action,
            all_required_params_present=response.query_result.all_required_params_present,
            query_text=response.query_result.query_text,
            name=response.query_result.intent.name,
            output_contexts=[c.name for c in response.query_result.output_contexts],
            input_contexts=contexts_var.get(),
        )
        logger.info(f"Detected intent: {intent}")
        return intent


async def translate(source: str, target: str, text: str):
    with create_channel('translate.googleapis.com') as channel:
        client = TranslationServiceStub(channel)
        request = TranslateTextRequest(
            contents=[text],
            source_language_code=source,
            target_language_code=target,
            parent=f'projects/{get_credentials().project_id}',
        )
        response: TranslateTextResponse = await client.TranslateText(request, metadata=await get_metadata())
        result = html.unescape(response.translations[0].translated_text)
        logger.debug(f"Translated: '{text}' -> '{result}'")
        return result


async def read_from_stream(stream):
    async for reply in stream:
        for chunk in reply.results:
            logger.debug(f"STT: {'|'.join(alt.transcript for alt in chunk.alternatives)}. Final={chunk.is_final}")
            yield chunk.alternatives
            if chunk.is_final:
                return
    else:
        raise EmptyUtterance


async def write_to_stream(stream, speech_stream, sent: asyncio.Event):
    try:
        async for speech in speech_stream:
            speech = speech.to_mono()
            if not sent.is_set():
                config = RecognitionConfig(
                    language_code=get_lang(),
                    audio_channel_count=1,
                    encoding=LINEAR16,
                    sample_rate_hertz=speech.rate,
                    enable_automatic_punctuation=True,
                )
                streaming_config = StreamingRecognitionConfig(config=config, interim_results=True)
                await stream.send_message(StreamingRecognizeRequest(streaming_config=streaming_config))
                sent.set()
            await stream.send_message(StreamingRecognizeRequest(audio_content=speech.data))
    finally:
        with suppress(ProtocolError):
            await stream.end()


@lru_cache()
def get_credentials():
    return Credentials.from_service_account_file(
        os.getenv('GOOGLE_APPLICATION_CREDENTIALS'),
        scopes=['https://www.googleapis.com/auth/cloud-platform', 'https://www.googleapis.com/auth/cloud-translation'],
    )


async def get_metadata():
    credentials = get_credentials()
    async with aiohttp.ClientSession() as sess:
        await credentials.refresh(Request(sess))
        return dict(authorization=f'Bearer {credentials.token}')


@contextmanager
def create_channel(host):
    channel = Channel(host, 443, ssl=True)
    try:
        yield channel
    finally:
        channel.close()


async def speech_stream_to_text(speech_stream):
    with create_channel('speech.googleapis.com') as channel:
        stub = SpeechStub(channel)
        sent = asyncio.Event()
        async with stub.StreamingRecognize.open(metadata=await get_metadata()) as stream:
            async with background_task(write_to_stream(stream, speech_stream, sent)):
                await sent.wait()
                async for text in read_from_stream(stream):
                    yield text
