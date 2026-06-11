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
import pycountry
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


def get_language_name(code: str) -> str:
    """Return the language name for an ISO 639-1 code, falling back to the code itself."""
    lang = pycountry.languages.get(alpha_2=code)
    return lang.name if lang is not None else code


class DecoderOnlyAttention(BaseStreamAtt):
    """
    Generic Decoder-only Attention-based policy for SpeechLLMs.

    The class handles:
       - Raw-waveform history accumulation.
       - Greedy generation with ``output_attentions=True``.
       - Building the proxy cross-attention matrix from self-attention weights.
        - Applying the StreamAtt-based policy on the proxy cross-attention matrix.

    The derived class should implement the following methods:
        - **load_model**: Loads the model and processor.
        - **build_prompt**: Builds the text prompt to use with audio inputs.
        - **build_processor_inputs**: Builds model inputs from the rolling audio history.
        - **_generate**: Generates tokens and proxy cross-attention scores.
        - **tokens_to_string**: Converts decoded tokens to a plain string.

    Args:
       config (SimpleNamespace): Configuration object. The following additional attributes are
           expected:
           - **attn_layer (int)**: Layer from which to extract attention scores. Defaults to 0.
           - **attn_head (int)**: Attention head to use. If not set, attention scores are averaged
             over all heads.
           - **average_attn_over_layers (bool)**: Whether to average the selected attention view
             over all decoder layers. Defaults to True.
           - **audio_history_max_duration (int)**: Maximum raw waveform length to keep in the
             rolling history, in seconds. Defaults to 180.
           - **max_new_tokens (int)**: Maximum tokens to generate per chunk. Defaults to 32.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.cross_attn_layer = getattr(self.config, "attn_layer", 0)
        self.cross_attn_head = getattr(self.config, "attn_head", None)
        self.average_attn_over_layers = getattr(self.config, "average_attn_over_layers", True)
        self.audio_history_max_duration = getattr(self.config, "audio_history_max_duration", 180)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = getattr(self.config, "max_new_tokens", 32)

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
        Build processor inputs from the entire rolling waveform history (float32, 16 kHz).

        The returned tensors must already be on ``self.device``.
        """
        ...

    @abstractmethod
    def _generate(self, waveform: np.ndarray) -> Tuple[List[str], torch.Tensor]:
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

    def build_raw_text_prefix(self) -> str:
        return "".join(self.text_history) if self.text_history else ""

    def _select_attn_from_layer(self, layer_attn: torch.Tensor) -> torch.Tensor:
        # Generation runs one stream at a time, so remove the singleton batch dimension.
        layer_attn = layer_attn.squeeze(0)
        if self.cross_attn_head is None:
            # Default behavior: average over all heads for this layer.
            return layer_attn.mean(dim=0)

        num_heads = layer_attn.shape[0]
        if self.cross_attn_head < 0 or self.cross_attn_head >= num_heads:
            raise ValueError(
                f"Invalid attn_head={self.cross_attn_head}. Layer has {num_heads} heads."
            )
        return layer_attn[self.cross_attn_head]

    def average_attn(self, attn) -> torch.Tensor:
        """
        Average or select attentions according to ``attn_layer``, ``attn_head``, and
        ``average_attn_over_layers``.

        If ``attn_head`` is not set, attention is averaged over heads. If
        ``average_attn_over_layers`` is set, the selected per-layer attention view is also averaged
        across layers; otherwise only ``attn_layer`` is used.
        """
        if self.average_attn_over_layers:
            # Average the per-layer attention view selected by _select_attn_from_layer.
            return torch.stack(
                [self._select_attn_from_layer(layer_attn) for layer_attn in attn],
                dim=0,
            ).mean(dim=0)
        return self._select_attn_from_layer(attn[self.cross_attn_layer])

    def _preprocess(self, waveform: np.float32) -> np.ndarray:
        """
        Append *waveform* to ``self.audio_history`` and enforce the maximum length.
        """
        if self.audio_history is None:
            self.audio_history = waveform
        else:
            self.audio_history = np.concatenate([self.audio_history, waveform])

        if len(self.audio_history) > self.audio_max_len:
            logger.warning("Audio history exceeded %d samples; trimming.", self.audio_max_len)
            self.audio_history = self.audio_history[-self.audio_max_len:]

        return self.audio_history
