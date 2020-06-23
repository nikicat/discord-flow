import audioop
import io
import logging
import os
import wave

import dialogflow_v2.types
from discord.opus import Decoder, Encoder
from google.cloud import texttospeech, speech_v1p1beta1

from .utils import sync_to_async, SAMPLE_WIDTH

logger = logging.getLogger(__name__)


async def text_to_speech(text):
    voice = texttospeech.VoiceSelectionParams(
        language_code='ru_RU', ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=Encoder.SAMPLING_RATE,
    )
    response = await sync_to_async(
        client.synthesize_speech, input=synthesis_input, voice=voice, audio_config=audio_config,
    )
    data = io.BytesIO(response.audio_content)
    audio = wave.open(data, 'rb')
    frames = audio.readframes(9999999)
    if False:
        converted, _ = audioop.ratecv(
            frames, audio.getsampwidth(), audio.getnchannels(), audio.getframerate(), Encoder.SAMPLING_RATE,
            None,
        )
    return audioop.tostereo(frames, audio.getsampwidth(), 1, 1)


async def speech_to_text(audio):
    client = speech_v1p1beta1.SpeechClient()

    # The language of the supplied audio
    language_code = "ru-RU"

    # Sample rate in Hertz of the audio data sent
    sample_rate_hertz = Decoder.SAMPLING_RATE

    # TODO: replace with AUDIO_ENCODING_OGG_OPUS
    encoding = speech_v1p1beta1.enums.RecognitionConfig.AudioEncoding.LINEAR16
    config = {
        "language_code": language_code,
        "sample_rate_hertz": sample_rate_hertz,
        "encoding": encoding,
    }
    mono = audioop.tomono(audio, SAMPLE_WIDTH, 0.5, 0.5)
    audio = {"content": mono}

    response = await sync_to_async(client.recognize, config, audio)
    transcript = response.results[0].alternatives[0].transcript
    logger.debug(f"Google STT result: {response} {transcript}")
    return transcript


async def detect_intent(utterance: str, session_id: str):
    dialogflow_project_id = os.getenv('DIALOGFLOW_PROJECT_ID')
    client = dialogflow_v2.SessionsClient()
    session = client.session_path(dialogflow_project_id, session_id)
    logger.debug(f"Session: {session}")
    if isinstance(utterance, str):
        query_input = dialogflow_v2.types.QueryInput(text=dialogflow_v2.types.TextInput(
            text=utterance, language_code='ru',
        ))
        kwargs = {}
    elif isinstance(utterance, bytes):
        query_input = dialogflow_v2.types.QueryInput(audio_config=dialogflow_v2.types.InputAudioConfig(
            audio_encoding=client.enums.AudioEncoding.AUDIO_ENCODING_LINEAR_16,
            sample_rate_hertz=Decoder.SAMPLING_RATE,
            language_code='ru',
            enable_word_info=True,
        ))
        kwargs = dict(input_audio=audioop.tomono(utterance, SAMPLE_WIDTH, 0.5, 0.5))
    else:
        raise ValueError(f"utterance should be audio (bytes) or text (str), not {type(utterance)}")

    response = await sync_to_async(
        client.detect_intent,
        session=session,
        query_input=query_input,
        **kwargs,
    )
    return response
