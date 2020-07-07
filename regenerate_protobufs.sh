#!/bin/sh -xe

python -m grpc_tools.protoc -I../googleapis --python_out=. --grpclib_python_out=. \
    google/api/{http,annotations,field_behavior,resource,client}.proto \
    google/cloud/speech/v1/cloud_speech.proto \
    google/cloud/texttospeech/v1/cloud_tts.proto \
    google/cloud/translate/v3/translation_service.proto \
    google/cloud/dialogflow/v2/{audio_config,context,entity_type,environment,intent,session_entity_type,session,validation_result,webhook}.proto \
    google/longrunning/operations.proto \
    google/rpc/status.proto \
    google/type/latlng.proto
