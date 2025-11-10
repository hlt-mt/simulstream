# Copyright 2025 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import unittest
from types import SimpleNamespace
from unittest.mock import patch


from simulstream.server.speech_processors.simuleval_wrapper import SimulEvalWrapper  # noqa: E402
from simulstream.server.speech_processors import IncrementalOutput  # noqa: E402


class DummyAgent:
    target_type = "text"

    def __init__(self, cfg):
        self.states = type("DummyStates", (), {})()
        self.tgt_lang = "en"


class TestSimulEvalWrapper(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            simuleval_agent="uts.speech_processors.test_simuleval_wrapper.DummyAgent",
            speech_chunk_size=1,
            latency_unit="word",
        )

    def _make_wrapper(self):
        """Helper to create a wrapper with necessary mocks."""
        with (
            patch("simulstream.server.speech_processors.simuleval_wrapper.torch") as mock_torch,
            patch("simulstream.server.speech_processors.simuleval_wrapper.get_detokenizer",
                  return_value=lambda t: "".join(t)),
            patch("simulstream.server.speech_processors.simuleval_wrapper."
                  "SimulEvalWrapper._segment_type_class", str),
        ):
            mock_torch.cuda.is_available.return_value = False
            return SimulEvalWrapper(self.config)

    def test_build_incremental_outputs_word(self):
        wrapper = self._make_wrapper()
        wrapper.latency_unit = "word"
        out = wrapper._build_incremental_outputs("Tornato a New York, sono")

        self.assertIsInstance(out, IncrementalOutput)
        self.assertEqual(out.new_tokens, ["Tornato", "a", "New", "York,", "sono"])
        self.assertEqual(out.new_string, "Tornato a New York, sono")
        self.assertEqual(out.deleted_tokens, [])
        self.assertEqual(out.deleted_string, "")

        out = wrapper._build_incremental_outputs("il capo dello sviluppo")

        self.assertIsInstance(out, IncrementalOutput)
        self.assertEqual(out.new_tokens, ["il", "capo", "dello", "sviluppo"])
        self.assertEqual(out.new_string, " il capo dello sviluppo")
        self.assertEqual(out.deleted_tokens, [])
        self.assertEqual(out.deleted_string, "")

    def test_build_incremental_outputs_start_char(self):
        wrapper = self._make_wrapper()
        wrapper.latency_unit = "char"
        out = wrapper._build_incremental_outputs('回到纽约后，我')
        self.assertEqual(out.new_tokens, ['回', '到', '纽', '约', '后', '，', '我'])

        out = wrapper._build_incremental_outputs('担任开发主管。')
        self.assertEqual(out.new_tokens, ['担', '任', '开', '发', '主', '管', '。'])

    def test_build_incremental_outputs_invalid(self):
        wrapper = self._make_wrapper()
        wrapper.latency_unit = "frame"
        with self.assertRaises(NotImplementedError):
            wrapper._build_incremental_outputs("something")


if __name__ == "__main__":
    unittest.main()
