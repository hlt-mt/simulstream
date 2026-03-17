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

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from simulstream.inference import process_audio
from simulstream.server.message_processor import MessageProcessor
from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.incremental_output import IncrementalOutput


def make_speech_processor(chunk_size_seconds=1.0):
    """Creates a mock SpeechProcessor with the minimal interface needed."""
    mock_output = IncrementalOutput(
        new_tokens=[], deleted_tokens=0, new_string="", deleted_string="")
    processor = MagicMock(
        spec=["speech_chunk_size", "process_chunk", "end_of_stream", "clear", "tokens_to_string"])
    processor.speech_chunk_size = chunk_size_seconds
    processor.process_chunk.return_value = mock_output
    processor.end_of_stream.return_value = mock_output
    processor.tokens_to_string.return_value = ""
    return processor


def make_message_processor(chunk_size_seconds=1.0):
    speech_processor = make_speech_processor(chunk_size_seconds)
    return MessageProcessor(client_id=0, speech_processor=speech_processor)


class TestProcessAudio(unittest.TestCase):

    def setUp(self):
        # Suppress metrics logging side effects
        patcher = patch("simulstream.server.message_processor.METRICS_LOGGER")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_exact_multiple(self):
        chunk_size = 1.0
        message_processor = make_message_processor(chunk_size)
        # 2 Full chunks, no reminder
        data = np.zeros(SAMPLE_RATE * 2, dtype=np.int16)

        process_audio(message_processor, SAMPLE_RATE, data)

        self.assertEqual(message_processor.speech_processor.process_chunk.call_count, 2)

    def test_remainder_chunk_not_sent_twice(self):
        chunk_size = 1.0
        message_processor = make_message_processor(chunk_size)
        # 2 Full chunks + a remainder of 0.5s
        data = np.zeros(int(SAMPLE_RATE * 2.5), dtype=np.int16)

        process_audio(message_processor, SAMPLE_RATE, data)

        # Process_chunk processes full chunks only; remainder stays buffered for end_of_stream
        self.assertEqual(message_processor.speech_processor.process_chunk.call_count, 2)
        self.assertEqual(len(message_processor.client_buffer), SAMPLE_RATE * 0.5)

    def test_single_chunk(self):
        chunk_size = 1.0
        message_processor = make_message_processor(chunk_size)
        # Data smaller than one chunk (process_chunk not called, data stays buffered)
        data = np.zeros(SAMPLE_RATE // 2, dtype=np.int16)  # 0.5s

        process_audio(message_processor, SAMPLE_RATE, data)

        message_processor.speech_processor.process_chunk.assert_not_called()
        self.assertEqual(len(message_processor.client_buffer), SAMPLE_RATE * 0.5)

    def test_empty_data(self):
        message_processor = make_message_processor()
        # Empty array (process_chunk never called, buffer remains empty)
        data = np.array([], dtype=np.int16)

        process_audio(message_processor, SAMPLE_RATE, data)

        message_processor.speech_processor.process_chunk.assert_not_called()
        self.assertEqual(message_processor.client_buffer, b'')


if __name__ == "__main__":
    unittest.main()
