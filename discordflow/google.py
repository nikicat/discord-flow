import logging
import os
from dataclasses import dataclass
from typing import Dict

import dialogflow_v2.types
from google.cloud import texttospeech, speech_v1p1beta1

from .utils import sync_to_async, Audio

logger = logging.getLogger(__name__)


async def text_to_speech(text) -> Audio:
    voice = texttospeech.VoiceSelectionParams(
        language_code='ru_RU', ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    rate = 48000
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=rate,
    )
    response = await sync_to_async(
        client.synthesize_speech, input=synthesis_input, voice=voice, audio_config=audio_config,
    )
    return Audio.loads(response.audio_content)


async def speech_to_text(audio: Audio):
    client = speech_v1p1beta1.SpeechClient()
    language_code = "ru-RU"

    # TODO: replace with AUDIO_ENCODING_OGG_OPUS
    encoding = speech_v1p1beta1.enums.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
        "language_code": language_code,
        "sample_rate_hertz": audio.rate,
        "encoding": encoding,
    }
    audio = {"content": audio.to_mono().data}

    response = await sync_to_async(client.recognize, config, audio)
    transcript = response.results[0].alternatives[0].transcript
    logger.info(f"STT: {response} {transcript}")
    return transcript


@dataclass
class Intent:
    text: str
    parameters: Dict[str, str]
    action: str
    all_required_params_present: bool


async def detect_intent(utterance: str, session_id: str) -> Intent:
    dialogflow_project_id = os.getenv('DIALOGFLOW_PROJECT_ID')
    client = dialogflow_v2.SessionsClient()
    session = client.session_path(dialogflow_project_id, session_id)
    logger.debug(f"Session: {session}")
    if isinstance(utterance, str):
        query_input = dialogflow_v2.types.QueryInput(text=dialogflow_v2.types.TextInput(
            text=utterance, language_code='ru',
        ))
        kwargs = {}
    elif isinstance(utterance, Audio):
        query_input = dialogflow_v2.types.QueryInput(audio_config=dialogflow_v2.types.InputAudioConfig(
            audio_encoding=client.enums.AudioEncoding.AUDIO_ENCODING_LINEAR_16,
            sample_rate_hertz=utterance.rate,
            language_code='ru',
            enable_word_info=True,
        ))
        kwargs = dict(input_audio=utterance.to_mono().data)
    else:
        raise ValueError(f"utterance should be Audio or text (str), not {type(utterance)}")

    response = await sync_to_async(
        client.detect_intent,
        session=session,
        query_input=query_input,
        **kwargs,
    )
    return Intent(
        text=response.query_result.filfillment_text,
        parameters={field.name: field.string_value for field in response.query_result.parameters.fields},
        action=response.query_result.action,
        all_required_params_present=response.query_result.all_required_params_present,
    )
