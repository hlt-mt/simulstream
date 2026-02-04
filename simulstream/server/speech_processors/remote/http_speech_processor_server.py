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
import base64
import json
import time
import logging
from functools import partial
from http import HTTPStatus
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from queue import Queue
import threading
from types import SimpleNamespace
from typing import Dict, Any, Optional

import numpy as np

import simulstream
from simulstream.config import yaml_config
from simulstream.server.speech_processors import build_speech_processor, SpeechProcessor


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGER = logging.getLogger(
    'simulstream.server.speech_processors.http.http_speech_processor_server')


class SpeechProcessorSessionManager:
    def __init__(self, speech_processor_config: SimpleNamespace, size: int, ttl: float):
        """
        Args:
            speech_processor_config: Configuration of the speech processors to create.
            size: How many speech processors to use.
            ttl: How long a session may stay idle before cleanup (in seconds).
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
                LOGGER.info(
                    f"Speech processor allocated to {session_id}, speech processors available: "
                    f"{self.available.qsize()}")
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
                LOGGER.info(
                    f"Session {session_id} closed, speech processors available: "
                    f"{self.available.qsize()}")
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
        self._cleanup_stop_event.set()
        self._cleanup_thread.join()


class HttpSpeechProcessorHandler(BaseHTTPRequestHandler):
    def __init__(
            self, *args, speech_processor_manager: SpeechProcessorSessionManager = None, **kwargs):
        self.speech_processor_manager = speech_processor_manager
        super().__init__(*args, **kwargs)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        data = self.rfile.read(length)
        return json.loads(data)

    def _send_json_response(self, code: int, message: Optional[Dict[str, Any]] = None):
        self.send_response(code)
        self.send_header("Content-type", "application/json; charset=utf-8")
        self.end_headers()
        if message is not None:
            self.wfile.write(json.dumps(message).encode("utf-8"))
        else:
            self.wfile.write("".encode("utf-8"))

    def do_GET(self):
        function_handler = getattr(self, "get_" + self.path.strip("/"))
        function_handler(**self._read_json())

    def do_POST(self):
        function_handler = getattr(self, "post_" + self.path.strip("/"))
        function_handler(**self._read_json())

    def do_PUT(self):
        function_handler = getattr(self, "put_" + self.path.strip("/"))
        function_handler(**self._read_json())

    def get_speech_chunk_size(self, session_id):
        processor = self.speech_processor_manager.get(session_id)
        self._send_json_response(HTTPStatus.OK, {"speech_chunk_size": processor.speech_chunk_size})

    def post_process_chunk(self, session_id, waveform):
        processor = self.speech_processor_manager.get(session_id)
        output = processor.process_chunk(
            np.frombuffer(base64.b64decode(waveform), dtype=np.float32))
        self._send_json_response(HTTPStatus.OK, {
            "new_tokens": output.new_tokens,
            "new_string": output.new_string,
            "deleted_tokens": output.deleted_tokens,
            "deleted_string": output.deleted_string,
        })

    def put_source_language(self, session_id, language):
        processor = self.speech_processor_manager.get(session_id)
        processor.set_source_language(language)
        self._send_json_response(HTTPStatus.NO_CONTENT)

    def put_target_language(self, session_id, language):
        processor = self.speech_processor_manager.get(session_id)
        processor.set_target_language(language)
        self._send_json_response(HTTPStatus.NO_CONTENT)

    def post_end_of_stream(self, session_id):
        processor = self.speech_processor_manager.get(session_id)
        output = processor.end_of_stream()
        self._send_json_response(HTTPStatus.OK, {
            "new_tokens": output.new_tokens,
            "new_string": output.new_string,
            "deleted_tokens": output.deleted_tokens,
            "deleted_string": output.deleted_string,
        })

    def post_clear(self, session_id):
        if self.speech_processor_manager.is_active(session_id):
            self.speech_processor_manager.close_session(session_id)
        self._send_json_response(HTTPStatus.NO_CONTENT)

    def get_tokens_to_string(self, session_id, tokens):
        processor = self.speech_processor_manager.get(session_id)
        output = processor.tokens_to_string(tokens)
        self._send_json_response(HTTPStatus.OK, {"tokens_as_string": output})


def serve(args: argparse.Namespace):
    LOGGER.info(f"Loading server configuration from {args.server_config}")
    server_config = yaml_config(args.server_config)
    LOGGER.info(f"Loading speech processor from {args.speech_processor_config}")
    speech_processor_loading_time = time.time()
    speech_processor_session_manager = SpeechProcessorSessionManager(
        yaml_config(args.speech_processor_config), server_config.pool_size, server_config.ttl
    )
    speech_processor_loading_time = time.time() - speech_processor_loading_time
    LOGGER.info(f"Loaded speech processor in {speech_processor_loading_time:.3f} seconds")

    custom_handler = partial(
        HttpSpeechProcessorHandler, speech_processor_manager=speech_processor_session_manager)
    httpd = ThreadingHTTPServer((server_config.hostname, server_config.port), custom_handler)
    LOGGER.info(f"Serving on http://{server_config.hostname}:{server_config.port}")
    httpd.serve_forever()
    speech_processor_session_manager.shutdown()


def main():
    LOGGER.info(f"HTTP speech processor server version: {simulstream.__version__}")
    parser = argparse.ArgumentParser("http_speech_processor_server")
    parser.add_argument("--server-config", type=str, default="config/http_server_example.yaml")
    parser.add_argument("--speech-processor-config", type=str, required=True)
    args = parser.parse_args()
    serve(args)


if __name__ == "__main__":
    main()
