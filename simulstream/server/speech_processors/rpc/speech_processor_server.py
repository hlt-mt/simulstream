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

import argparse
import time
from concurrent import futures
import logging
from queue import Queue
import threading
from types import SimpleNamespace

import grpc
import numpy as np

import simulstream
from simulstream.config import yaml_config
from simulstream.server.speech_processors import build_speech_processor, SpeechProcessor
from simulstream.server.speech_processors.rpc import speech_processor_pb2_grpc as sp_pb2_grpc
from simulstream.server.speech_processors.rpc import speech_processor_pb2 as sp_pb2
from simulstream.server.speech_processors.rpc.speech_processor_pb2 \
    import google_dot_protobuf_dot_empty__pb2 as empty_pb


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGER = logging.getLogger('simulstream.server.speech_processors.rpc.speech_processor_server')


class SpeechProcessorSessionManager:
    def __init__(self, speech_processor_config: SimpleNamespace, size: int, ttl: float):
        """
        Args:
            speech_processor_config: Configuration of the speech processors to create.
            size: How many speech processors to use.
            ttl: How long a session may stay idle before cleanup in seconds.
        """
        self._sessions = {}
        self._last_access = {}
        self._lock = threading.Lock()
        self.size = size
        self.ttl = ttl
        self.available = Queue(maxsize=size)
        for _ in range(size):
            self.available.put_nowait(build_speech_processor(speech_processor_config))

        # starting cleanup loop
        self._cleanup_stop_event = threading.Event()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup,
            daemon=True,
        )
        self._cleanup_thread.start()

    def get(self, session_id) -> SpeechProcessor:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = self.available.get_nowait()
            self._last_access[session_id] = time.time()
            return self._sessions[session_id]

    def is_active(self, session_id) -> bool:
        with self._lock:
            return session_id in self._sessions

    def close_session(self, session_id):
        with self._lock:
            if session_id in self._sessions:
                speech_processor = self._sessions.pop(session_id)
                speech_processor.clear()
                self.available.put_nowait(speech_processor)
            if session_id in self._last_access:
                self._last_access.pop(session_id)

    def _cleanup(self):
        while not self._cleanup_stop_event.is_set():
            time.sleep(self.ttl)
            now = time.time()
            expired = []
            with self._lock:
                for session_id in self._sessions.keys():
                    if session_id not in self._last_access or \
                            now - self._last_access[session_id] > self.ttl:
                        expired.append(session_id)

            for session_id in expired:
                self.close_session(session_id)

    def shutdown(self) -> None:
        """
        Stop cleanup thread.
        """
        self._cleanup_stop_event.set()
        self._cleanup_thread.join()


class GrpcSpeechProcessorWrapper(sp_pb2_grpc.SpeechProcessorServiceServicer):
    def __init__(self, speech_processor_manager: SpeechProcessorSessionManager):
        self.speech_processor_manager = speech_processor_manager

    def SpeechChunkSize(self, request, context):
        processor = self.speech_processor_manager.get(request.session_id)
        return sp_pb2.SpeechChunkSizeData(speech_chunk_size=processor.speech_chunk_size)

    def ProcessChunk(self, request, context):
        processor = self.speech_processor_manager.get(request.session_id)
        waveform = np.frombuffer(request.pcm_f32, dtype=np.float32)
        output = processor.process_chunk(waveform)
        return sp_pb2.IncrementalOutput(
            new_tokens=output.new_tokens,
            new_string=output.new_string,
            deleted_tokens=output.deleted_tokens,
            deleted_string=output.deleted_string,
        )

    def SetSourceLanguage(self, request, context):
        processor = self.speech_processor_manager.get(request.session_id)
        processor.set_source_language(request.language)
        return empty_pb.Empty()

    def SetTargetLanguage(self, request, context):
        processor = self.speech_processor_manager.get(request.session_id)
        processor.set_target_language(request.language)
        return empty_pb.Empty()

    def EndOfStream(self, request, context):
        processor = self.speech_processor_manager.get(request.session_id)
        output = processor.end_of_stream()
        return sp_pb2.IncrementalOutput(
            new_tokens=output.new_tokens,
            new_string=output.new_string,
            deleted_tokens=output.deleted_tokens,
            deleted_string=output.deleted_string,
        )

    def Clear(self, request, context):
        if self.speech_processor_manager.is_active(request.session_id):
            self.speech_processor_manager.close_session(request.session_id)
        return empty_pb.Empty()

    def Tokens2String(self, request, context):
        processor = self.speech_processor_manager.get(request.session_id)
        output = processor.tokens_to_string(request.tokens)
        return sp_pb2.StringFromTokens(tokens_as_string=output)


def serve(args: argparse.Namespace):
    LOGGER.info(f"Loading server configuration from {args.server_config}")
    server_config = yaml_config(args.server_config)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=server_config.pool_size))
    LOGGER.info(f"Loading speech processor from {args.speech_processor_config}")
    speech_processor_loading_time = time.time()
    speech_processor_session_manager = SpeechProcessorSessionManager(
        yaml_config(args.speech_processor_config), server_config.pool_size, server_config.ttl
    )
    speech_processor_loading_time = time.time() - speech_processor_loading_time
    LOGGER.info(f"Loaded speech processor in {speech_processor_loading_time:.3f} seconds")

    sp_pb2_grpc.add_SpeechProcessorServiceServicer_to_server(
        GrpcSpeechProcessorWrapper(speech_processor_session_manager), server
    )

    server.add_insecure_port(f"{server_config.hostname}:{server_config.port}")
    server.start()
    LOGGER.info(f"gRPC server listening on {server_config.hostname}:{server_config.port}")
    server.wait_for_termination()
    speech_processor_session_manager.shutdown()


def main():
    LOGGER.info(f"gRPC speech processor server version: {simulstream.__version__}")
    parser = argparse.ArgumentParser("grpc_speech_processor_server")
    parser.add_argument("--server-config", type=str, default="config/grpc_server_example.yaml")
    parser.add_argument("--speech-processor-config", type=str, required=True)
    args = parser.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
