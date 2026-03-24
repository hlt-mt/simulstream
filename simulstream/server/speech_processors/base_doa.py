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

import logging
from abc import abstractmethod
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import torch

from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.base_streamatt import BaseStreamAtt

logger = logging.getLogger(__name__)


TEMPLATED_SPEECH_PROMPT = \
    ("You are a professional {src_lang}-to-{tgt_lang} translator. Your goal is to accurately "
     "convey the meaning and nuances of the original {src_lang} speech while adhering to "
     "{tgt_lang} grammar, vocabulary, and cultural sensitivities. Use precise terminology and a "
     "tone appropriate for academic or instructional materials. Produce only the {tgt_lang} "
     "translation, without any additional explanations or commentary. Please translate the "
     "provided {src_lang} speech into {tgt_lang}:")

LANG_MAPPER = {"en": "English", "it": "Italian"}


class DecoderOnlyAttention(BaseStreamAtt):
    """
    Generic Decoder-only Attention-based policy for SpeechLLMs.

    The class handles:
    - Rolling raw-waveform history accumulation.
    - Greedy generation with ``output_attentions=True``.
    - Building the proxy cross-attention matrix from self-attention weights.
    - Token decoding.

    Subclasses must implement the five abstract methods listed below.

    Parameters
    ----------
    config : SimpleNamespace
        All fields from :class:`BaseStreamAtt`, plus:

        device : str
            Torch device string.  Default: ``"cuda"``.
        audio_max_frames : int
            Maximum raw waveform samples to keep in the rolling history
            (at 16 kHz).  Default: ``480_000`` (30 s).
        max_new_tokens : int
            Maximum tokens to generate per chunk.  Default: ``200``.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.cross_attn_layer = getattr(self.config, "attention_layer", 3)
        self.max_new_tokens = getattr(self.config, "max_new_tokens", 4096)
        self.audio_history_max_duration = getattr(self.config, "audio_history_max_duration", 360)
        self.src_lang_tag = getattr(self.config, "src_lang_tag", "en")
        self.tgt_lang_tag = getattr(self.config, "tgt_lang_tag", "en")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def audio_max_len(self) -> int:
        """Maximum raw-waveform samples to keep in the rolling audio history."""
        return self.audio_history_max_duration * SAMPLE_RATE

    @abstractmethod
    def load_model(self, config: SimpleNamespace) -> None:
        """
        Load the model and processor from *config* and assign them to
        ``self.model`` and ``self.processor``.

        The model **must** be loaded with ``output_attentions=True`` (or the
        equivalent flag for the architecture) and
        ``_attn_implementation="eager"``.
        """
        ...

    @abstractmethod
    def build_prompt(self) -> str:
        """
        Return the prompt string to be used with audio tokens.
        """
        ...

    @abstractmethod
    def build_processor_inputs(self, waveform: np.ndarray) -> dict:
        """
        Given the *entire* rolling waveform history (float32, 16 kHz), return
        a ``dict`` of ``torch.Tensor`` inputs ready to be passed to
        ``self.model.generate(**inputs, …)``.

        The tensors must already be on ``self.device``.
        """
        ...

    @abstractmethod
    def _generate(self, inputs: dict) -> Tuple[List[str], torch.Tensor]:
        """
        Generate tokens from the given inputs together with the self-attention scores.

        Returns:
            Tuple[List[str], torch.Tensor]:
                List[str]: A list of generated tokens.
                torch.Tensor: Self-attention scores between speech and text with dimension
                (token_len, audio_len).
        """
        ...

    @abstractmethod
    def tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a list of decoded tokens to a plain output string."""
        ...

    def set_target_language(self, language: str) -> None:
        self.tgt_lang = language

    def set_source_language(self, language: str) -> None:
        self.src_lang = language

    def _preprocess(self, waveform: np.float32) -> dict:
        """
        Append *waveform* to ``self.audio_history``, enforce the maximum length,
        and delegate to :meth:`build_processor_inputs`.
        """
        if self.audio_history is None:
            self.audio_history = waveform
        else:
            self.audio_history = np.concatenate([self.audio_history, waveform])

        if len(self.audio_history) > self.audio_max_len:
            logger.warning("Audio history exceeded %d samples; trimming.", self.audio_max_len)
            self.audio_history = self.audio_history[-self.audio_max_len:]

        return self.build_processor_inputs(self.audio_history)
