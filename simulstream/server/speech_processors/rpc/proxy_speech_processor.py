# Copyright 2026 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

from typing import List
import uuid

import grpc
import numpy as np

from simulstream.server.speech_processors import SpeechProcessor, IncrementalOutput
from simulstream.server.speech_processors.rpc.speech_processor_pb2_grpc import SpeechProcessorServiceStub
from simulstream.server.speech_processors.rpc import speech_processor_pb2


class ProxyGrpcSpeechProcessor(SpeechProcessor):
    """
    gRPC-based proxy implementation of :class:`SpeechProcessor`.

    This class does not perform speech processing locally. Instead, it forwards
    all method calls to a remote speech processor exposed via gRPC, maintaining
    a dedicated session on the server side.

    Each instance of this class corresponds to exactly one remote session.
    """

    @classmethod
    def load_model(cls, config):
        pass

    def __init__(self, config):
        super().__init__(config)
        self.channel = grpc.insecure_channel(f"{config.hostname}:{config.port}")
        self.session_id = uuid.uuid4().hex
        self.stub = SpeechProcessorServiceStub(self.channel)

    @property
    def speech_chunk_size(self) -> float:
        print("sending speech chunk size req")
        response = self.stub.SpeechChunkSize(
            speech_processor_pb2.SpeechChunkSizeRequest(session_id=self.session_id))
        print("got speech chunk size reply")
        return response.speech_chunk_size

    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        response = self.stub.ProcessChunk(speech_processor_pb2.AudioChunk(
            session_id=self.session_id, pcm_f32=waveform.astype(np.float32).tobytes()))
        return IncrementalOutput(
            new_tokens=list(response.new_tokens),
            new_string=response.new_string,
            deleted_tokens=list(response.deleted_tokens),
            deleted_string=response.deleted_string
        )

    def set_source_language(self, language):
        self.stub.SetSourceLanguage(speech_processor_pb2.SetLanguageRequest(
            session_id=self.session_id, language=language))

    def set_target_language(self, language):
        self.stub.SetTargetLanguage(speech_processor_pb2.SetLanguageRequest(
            session_id=self.session_id, language=language))

    def end_of_stream(self) -> IncrementalOutput:
        response = self.stub.EndOfStream(
            speech_processor_pb2.EndOfStreamRequest(session_id=self.session_id))
        return IncrementalOutput(
            new_tokens=list(response.new_tokens),
            new_string=response.new_string,
            deleted_tokens=list(response.deleted_tokens),
            deleted_string=response.deleted_string
        )

    def clear(self):
        self.stub.Clear(speech_processor_pb2.ClearRequest(session_id=self.session_id))

    def tokens_to_string(self, tokens: List[str]) -> str:
        response = self.stub.Tokens2String(
            speech_processor_pb2.Tokens(session_id=self.session_id, tokens=tokens))
        return response.tokens_as_string
