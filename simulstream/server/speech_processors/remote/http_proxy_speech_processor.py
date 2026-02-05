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

import base64
import json
from http import HTTPStatus
from typing import List, Any, Dict, Optional
import uuid
import urllib.request

import numpy as np

from simulstream.server.speech_processors import SpeechProcessor, IncrementalOutput


class HttpProxySpeechProcessor(SpeechProcessor):
    """
    HTTP-based proxy implementation of :class:`SpeechProcessor`.

    This class does not perform speech processing locally. Instead, it forwards
    all method calls to a remote speech processor exposed via HTTP, maintaining
    a dedicated session on the server side.

    Each instance of this class corresponds to exactly one remote session.
    """

    @classmethod
    def load_model(cls, config):
        pass

    def __init__(self, config):
        super().__init__(config)
        self.base_url = f"http://{config.hostname}:{config.port}/"
        self.session_id = uuid.uuid4().hex
        self._cached_speech_chunk_size = None

    def _http_request(
            self, path: str, method: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(req) as resp:
            if resp.status == HTTPStatus.NO_CONTENT:
                return None
            return json.loads(resp.read())

    @staticmethod
    def _to_incremental_outputs(json_dict: Dict[str, Any]):
        return IncrementalOutput(
            new_tokens=json_dict["new_tokens"],
            new_string=json_dict["new_string"],
            deleted_tokens=json_dict["deleted_tokens"],
            deleted_string=json_dict["deleted_string"]
        )

    @property
    def speech_chunk_size(self) -> float:
        if self._cached_speech_chunk_size is None:
            response = self._http_request("speech_chunk_size", "GET", {
                "session_id": self.session_id
            })
            self._cached_speech_chunk_size = response["speech_chunk_size"]
        return self._cached_speech_chunk_size

    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        response = self._http_request("process_chunk", "POST", {
            "session_id": self.session_id,
            "waveform": base64.b64encode(waveform.tobytes()).decode("utf-8"),
        })
        return self._to_incremental_outputs(response)

    def set_source_language(self, language):
        self._http_request("source_language", "PUT", {
            "session_id": self.session_id,
            "language": language,
        })

    def set_target_language(self, language):
        self._http_request("target_language", "PUT", {
            "session_id": self.session_id,
            "language": language,
        })

    def end_of_stream(self) -> IncrementalOutput:
        response = self._http_request("end_of_stream", "POST", {
            "session_id": self.session_id,
        })
        return self._to_incremental_outputs(response)

    def clear(self):
        self._http_request("clear", "POST", {
            "session_id": self.session_id,
        })

    def tokens_to_string(self, tokens: List[str]) -> str:
        response = self._http_request("tokens_to_string", "GET", {
            "session_id": self.session_id,
            "tokens": tokens,
        })
        return response["tokens_as_string"]
