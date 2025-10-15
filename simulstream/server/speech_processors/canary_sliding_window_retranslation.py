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

from types import SimpleNamespace
from typing import List

import numpy as np
import torch
from nemo.collections.asr.models import ASRModel

from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.sliding_window_retranslation import \
    SlidingWindowRetranslator


class CanarySlidingWindowRetranslator(SlidingWindowRetranslator):
    """
    Perform Sliding Window Retranslation with Canary.
    """

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        if not hasattr(cls, "model") or cls.model is None:
            cls.model = ASRModel.from_pretrained(model_name=config.model_name)
            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            assert cls.model.preprocessor._sample_rate == SAMPLE_RATE
            cls.model.to(cls.device)

    def _generate(self, speech: torch.Tensor) -> List[str]:
        output = self.model.transcribe(
            speech, source_lang=self.src_lang_tag, target_lang=self.tgt_lang_tag)
        return self.model.tokenizer.ids_to_tokens(output[0].y_sequence)

    def _tokens_to_string(self, tokens: List[str]) -> str:
        # avoid that the initial space, if it is there, get removed in the detokenization
        check_for_init_space = self.text_history is not None and len(self.text_history) > 0
        if check_for_init_space:
            tokens = [' '] + tokens
        text = self.model.tokenizer.tokens_to_text(tokens)
        if check_for_init_space:
            text = text[1:]
        return text

    def _preprocess(self, waveform: np.float32) -> torch.Tensor:
        """
        Extracts the filter-bank features from the input waveform and appends them to the audio
        history. Returns the concatenated audio history and new frames, taking the last
        `self.window_len` frames, and returns it after storing it in the audio history.
        """
        if self.audio_history is not None:
            waveform = np.concatenate((self.audio_history, waveform))
        new_speech_len = len(waveform)
        if new_speech_len > self.window_len:
            waveform = waveform[-self.window_len:]
        self.audio_history = waveform
        return torch.tensor(waveform).to(self.device)

    def set_target_language(self, language: str) -> None:
        self.tgt_lang_tag = language

    def set_source_language(self, language: str) -> None:
        self.src_lang_tag = language
