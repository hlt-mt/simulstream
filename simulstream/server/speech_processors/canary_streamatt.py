# Copyright 2025 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import logging
import torch
import numpy as np

from types import SimpleNamespace
from typing import List, Tuple

import copy

from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.base_streamatt import BaseStreamAtt

from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.submodules.multitask_decoding import (
    MultiTaskDecodingConfig,
)
from nemo.collections.asr.models.aed_multitask_models import (
    MultiTaskTranscriptionConfig,
)

logger = logging.getLogger(__name__)


class CanaryStreamAtt(BaseStreamAtt):
    """
    StreamAtt policy implementation for NVIDIA's Canary-v2 model.

    Args:
        config (SimpleNamespace): Configuration object.
            Supported attributes:
            - **audio_history_max_duration (int)**: Maximum audio history in seconds.
              Defaults to ``30``.
            - **num_beams (int)**: Number of beams to use for beam search decoding.
              Defaults to ``5``.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self._audio_history_max_duration = getattr(self.config, "audio_history_max_duration", 30)

        expected_mel_hop_samples = (
            self.model.cfg.preprocessor.window_stride * self.model.cfg.preprocessor.sample_rate
        )

        assert self.mel_hop_samples == expected_mel_hop_samples, (
            f"mel_hop_samples is set to {self.mel_hop_samples} in the config, but the loaded "
            f"model's preprocessor uses {expected_mel_hop_samples} samples per mel frame"
        )

        # Build the transcription config, which is reused for every transcribe() call.
        self.transcription_cfg = MultiTaskTranscriptionConfig(
            batch_size=1,
            return_hypotheses=True,
            enable_chunking=False,
            verbose=False,
        )

    @property
    def audio_max_len(self) -> int:
        """Maximum audio history length in raw waveform samples."""
        return self._audio_history_max_duration * SAMPLE_RATE

    def set_source_language(self, language: str) -> None:
        self.src_lang = language

    def set_target_language(self, language: str) -> None:
        self.tgt_lang = language

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        if not hasattr(cls, "model") or cls.model is None:
            cls.model = ASRModel.from_pretrained(model_name=config.model_name)

            # Configure decoding strategy
            multitask_decoding = MultiTaskDecodingConfig()
            multitask_decoding.strategy = "beam"
            multitask_decoding.return_xattn_scores = True
            multitask_decoding.beam.beam_size = getattr(config, "num_beams", 5)
            cls.model.change_decoding_strategy(multitask_decoding)

            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            assert cls.model.cfg.preprocessor.sample_rate == SAMPLE_RATE
            cls.model.to(cls.device)

    def _build_transcription_config(self):
        """
        Return a ``MultiTaskTranscriptionConfig`` whose prompt encodes the current source/target
        languages, task, PNC preference, and forced decoder prefix.
        """

        default_turns = self.model.prompt.get_default_dialog_slots()
        default_slots = copy.deepcopy(default_turns[0]["slots"])
        default_slots["source_lang"] = self.src_lang
        default_slots["target_lang"] = self.tgt_lang

        turns = [
            {
                "role": "user", "slots": default_slots
            },
            {
                "role": "user_prefix",
                "slots": {
                    "prefix": self.model.tokenizer.tokens_to_text(self.text_history)
                },
            },
        ]

        cfg_copy = copy.deepcopy(self.transcription_cfg)
        cfg_copy.prompt = turns

        return cfg_copy

    def _preprocess(self, waveform: np.ndarray) -> np.ndarray:
        """
        Append the incoming waveform chunk to the raw audio history and return it.

        Returns:
            np.ndarray: Accumulated raw audio history.
        """
        waveform = waveform.astype(np.float32)
        if self.audio_history is None:
            self.audio_history = waveform
        else:
            self.audio_history = np.concatenate(
                [self.audio_history, waveform])

        return self.audio_history

    def _generate(self, speech: np.ndarray) -> Tuple[List[str], torch.Tensor]:
        override_config = self._build_transcription_config()

        with torch.inference_mode():
            output = self.model.transcribe(audio=speech, override_config=override_config)

        hypothesis = output[0]

        token_ids = hypothesis.y_sequence.detach().cpu().tolist()
        tokens = self.model.tokenizer.ids_to_tokens(token_ids)

        xatt_raw = hypothesis.xatt_scores[self.cross_attn_layer]
        xatt = xatt_raw.mean(dim=0).cpu()  # we average over heads
        xatt = self.normalize_attn(xatt)

        return tokens, xatt

    def tokens_to_string(self, tokens: List[str]) -> str:
        return self.model.tokenizer.tokens_to_text(tokens)
