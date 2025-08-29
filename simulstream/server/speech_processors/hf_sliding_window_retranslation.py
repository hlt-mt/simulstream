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
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.sliding_window_retranslation import \
    SlidingWindowRetranslator


class HFSlidingWindowRetranslator(SlidingWindowRetranslator):

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        if not hasattr(cls, "model") or cls.model is None:
            lang_tags = None
            if hasattr(config, "supported_langs") and config.supported_langs is not None:
                lang_tags = [
                    config.lang_tag_template.format(lang) for lang in config.supported_langs]
            cls.processor = AutoProcessor.from_pretrained(
                config.hf_model_name,
                additional_special_tokens=lang_tags)
            cls.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                config.hf_model_name, trust_remote_code=True)
            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls.model.to(cls.device)

    def _generate(self, speech: torch.Tensor) -> List[str]:
        speech_seconds = speech.shape[1] / 100  # 1 frame every 10 ms
        extra_kwargs = {
            "max_new_tokens": int(max(self.max_tokens_per_second * speech_seconds, 10))}
        if self.tgt_lang_tag is not None:
            extra_kwargs["forced_bos_token_id"] = self.tgt_lang_tag
        generated_ids = self.model.generate(speech, **extra_kwargs)[0]
        return self.processor.tokenizer.convert_ids_to_tokens(
            generated_ids, skip_special_tokens=True)

    def _tokens_to_string(self, tokens: List[str]) -> str:
        return self.processor.tokenizer.convert_tokens_to_string(tokens)

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
        new_speech = self.processor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt")["input_features"]
        return new_speech.to(self.device)

    def set_target_language(self, language: str) -> None:
        lang_tag_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.config.lang_tag_template.format(language))
        self.tgt_lang_tag = torch.tensor(lang_tag_id, dtype=torch.int, device=self.device)

    def set_source_language(self, language: str) -> None:
        pass
