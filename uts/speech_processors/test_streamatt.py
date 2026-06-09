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
import torch
import numpy as np
from typing import Dict, List, Tuple, Union

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


class FakeStreamAtt(BaseStreamAtt):

    def _preprocess(self, waveform: np.float32) -> Union[Dict[str, torch.Tensor], torch.Tensor]:
        raise NotImplementedError("_preprocess not implemented in FakeStreamAtt")

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        raise NotImplementedError("load_model not implemented in FakeStreamAtt")

    def set_source_language(self, language: str) -> None:
        pass

    def set_target_language(self, language: str) -> None:
        pass

    def tokens_to_string(self, tokens: List[str]) -> str:
        return " ".join(tokens)

    def _generate(self, speech: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        raise NotImplementedError("_generate not implemented in FakeStreamAtt")

    @property
    def audio_max_len(self) -> float:
        return 10000


class TestUpdateSpeechHistory(unittest.TestCase):
    def _run_update_speech_history(self, use_raw_audio_history):
        config = SimpleNamespace(
            use_raw_audio_history=use_raw_audio_history,
            audio_subsampling_factor=2,
            mel_hop_samples=2,
            text_history=SimpleNamespace(
                type="simulstream.server.speech_processors.base_streamatt.FixedWordsTextHistory",
            )

        )
        audio = np.arange(40, dtype=np.float32)
        proc = FakeStreamAtt(config)
        proc.text_history = ["▁hello"]
        proc.audio_history = audio.copy()

        attn = torch.zeros(2, 10)
        attn[1, 2] = 1.0

        proc._update_speech_history(discarded_text=1, cross_attn=attn)
        return proc.audio_history.tolist()

    def test_update_speech_history_trims_audio_with_raw_audio(self):
        audio_hist = self._run_update_speech_history(use_raw_audio_history=True)
        # 2 audio token discarded, subsampling factor is 2,
        # num mel hop is 2, so 2*2*2=8 samples removed
        self.assertListEqual(audio_hist, list(np.arange(8, 40, dtype=np.float32)))

    def test_update_speech_history_trims_audio(self):
        audio_hist = self._run_update_speech_history(use_raw_audio_history=False)
        # 2 audio token discarded, subsampling factor is 2, so 2*2=4 samples removed
        self.assertListEqual(audio_hist, list(np.arange(4, 40, dtype=np.float32)))


if __name__ == "__main__":
    unittest.main()
