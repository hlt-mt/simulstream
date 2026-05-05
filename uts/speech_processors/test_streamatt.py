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
from types import SimpleNamespace
from unittest.mock import MagicMock
import torch
import numpy as np


from simulstream.server.speech_processors.base_streamatt import (
    BaseStreamAtt,
    PunctuationTextHistory,
)


class TestPunctuationTextHistory(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace()
        self.punctuation_text_history = PunctuationTextHistory(self.config)

    def test_punctuation_last(self):
        """ Test PunctuationTextHistory method when the history ends with strong punctuation. """
        # Test word level
        en_history = ["Hi", "!", "I", "am", "Sara", "."]
        selected_history = self.punctuation_text_history.select_text_history(en_history)
        self.assertEqual(selected_history, ["I", "am", "Sara", "."])

        # Test character level
        zh_history = ['担', '任', '开', '发', '主', '管', '。']
        selected_history = self.punctuation_text_history.select_text_history(zh_history)
        self.assertEqual(selected_history, ['担', '任', '开', '发', '主', '管', '。'])

    def test_punctuation_in_between(self):
        """ Test PunctuationTextHistory method when punctuation separates two sentences. """
        # Test word level
        en_history = ["Hi", "!", "I", "am", "Sara"]
        selected_history = self.punctuation_text_history.select_text_history(en_history)
        self.assertEqual(selected_history, ["I", "am", "Sara"])

        # Test character level
        zh_history = ['开', '发', '主', '管', '。', '担', '任']
        selected_history = self.punctuation_text_history.select_text_history(zh_history)
        self.assertEqual(selected_history, ['担', '任'])

    def test_no_strong_punctuation(self):
        """ Test PunctuationTextHistory method when no strong punctuation is present. """
        # Test word level
        en_history = ["Hi", ",", "I", "am", "Sara"]
        selected_history = self.punctuation_text_history.select_text_history(en_history)
        self.assertEqual(selected_history, ["Hi", ",", "I", "am", "Sara"])

        # Test character level
        zh_history = ['回', '到', '纽', '约', '后', '，', '我']
        selected_history = self.punctuation_text_history.select_text_history(zh_history)
        self.assertEqual(selected_history, ['回', '到', '纽', '约', '后', '，', '我'])


def _make_mock(text_history, audio_history, frames_to_audio_history, audio_max_len=100_000):
    proc = MagicMock()
    proc.text_history = text_history
    proc.audio_history = audio_history
    proc.frames_to_audio_history = frames_to_audio_history
    proc.audio_max_len = audio_max_len
    proc._cut_audio_exceeding_maxlen.side_effect = \
        lambda: BaseStreamAtt._cut_audio_exceeding_maxlen(proc)
    return proc


def _cross_attn(n_text_tokens, n_audio_frames, earliest_attended_frame, discarded_text=0):
    attn = torch.zeros(discarded_text + n_text_tokens, n_audio_frames)
    for i in range(discarded_text, discarded_text + n_text_tokens):
        attn[i, earliest_attended_frame] = 1.0
    return attn


class TestUpdateSpeechHistory(unittest.TestCase):
    def test_trim_audio_history(self):
        """ Test that audio history is trimmed correctly """
        audio = np.arange(40, dtype=np.float32)
        proc = _make_mock(["▁hello"], audio.copy(), frames_to_audio_history=4)
        attn = _cross_attn(
            n_text_tokens=1, n_audio_frames=10, earliest_attended_frame=2, discarded_text=1)
        BaseStreamAtt._update_speech_history(proc, discarded_text=1, cross_attn=attn)
        np.testing.assert_array_equal(proc.audio_history, audio[8:])


if __name__ == "__main__":
    unittest.main()
