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
from typing import List, Tuple, Any, Dict

import numpy as np
import pycountry
import torch

from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.base_streamatt import BaseStreamAtt


logger = logging.getLogger(__name__)


def get_language_name(code: str) -> str:
    """Return the language name for an ISO 639-1 code, falling back to the code itself."""
    lang = pycountry.languages.get(alpha_2=code)
    if lang is not None:
        return lang.name
    else:
        logger.warning(f"Language code '{code}' not found in the language list. Using language "
                       f"code directly in the prompt, but this can lead to unexpected behavior.")
        return code


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
        - **_do_generate**: Returns newly-generated tokens and self-attention scores.
        - **_find_audio_positions**: Returns the indices of audio tokens.

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
        self.prompt = getattr(self.config, "prompt", "Translate the audio to {tgt_lang}:")
        logger.debug("Prompt:\n%s", self.prompt)

    @property
    def audio_max_len(self) -> int:
        """Maximum raw-waveform samples to keep in the rolling audio history."""
        return self.audio_history_max_duration * SAMPLE_RATE

    @abstractmethod
    def load_model(self, config: SimpleNamespace) -> None:
        """
        Load the model and processor from *config* and assign them to ``self.model`` and
        ``self.processor``.

        The model **must** be loaded with ``output_attentions=True`` (or the equivalent flag for
        the architecture) and ``_attn_implementation="eager"``.
        """
        ...

    @abstractmethod
    def build_prompt(self) -> str:
        """
        Return the prompt string to be used with audio tokens.
        """
        ...

    @abstractmethod
    def build_processor_inputs(self, waveform: np.ndarray) -> Any:
        """
        Build processor inputs from the entire rolling waveform history (float32, 16 kHz).
        """
        ...

    @abstractmethod
    def _do_generate(self, inputs: Dict[str, Any]) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Runs the actual generation from the underlying model and returns the generated tokens and
        the corresponding self-attention scores.
        """
        ...

    @abstractmethod
    def _find_audio_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token positions corresponding to the encoded audio span."""
        ...

    def _generate(self, waveform: np.ndarray) -> Tuple[List[str], torch.Tensor]:
        """
        Generate tokens from the given inputs together with the self-attention scores.

        Returns:
            Tuple[List[str], torch.Tensor]:
                List[str]: A list of generated tokens.
                torch.Tensor: Self-attention scores between speech and text with dimension
                (token_len, audio_len).
        """
        inputs = self.build_processor_inputs(waveform).to(self.device)
        input_ids = inputs["input_ids"]  # (1, input_len)
        input_len = input_ids.shape[1]

        audio_positions = self._find_audio_positions(input_ids)
        audio_len = audio_positions.shape[0]

        # Run the actual generate on the underlying model
        new_tokens, attentions = self._do_generate(inputs)

        # Build proxy cross-attention for the hypothesis (prefix + new_tokens)
        prefill_attn = self.average_attn(attentions[0])
        prefix_len = len(self.text_history) if self.text_history else 0
        if prefix_len > 0:
            # Prefix rows come from the prefill pass
            prefix_rows = prefill_attn[input_len - prefix_len:, :][:, audio_positions]
        else:
            prefix_rows = torch.zeros(0, audio_len, device=self.device)

        if new_tokens:
            # The prefill pass predicts the first generated token, so its last row corresponds to
            # the first generated token's proxy audio-attention
            first_new_row = prefill_attn[-1:, audio_positions]
            # Other tokens' attention is present in each generation step
            new_rows = [
                self.average_attn(step_attn).squeeze(0)[audio_positions]
                for step_attn in attentions[1:]
            ]
            subsequent_new_attn = torch.stack(new_rows, dim=0)
            new_attn = torch.cat([first_new_row, subsequent_new_attn], dim=0)
        else:
            new_attn = torch.zeros(0, audio_len, device=self.device)

        cross_attn = torch.cat([prefix_rows, new_attn], dim=0)
        cross_attn = self.normalize_attn(cross_attn)
        return new_tokens, cross_attn

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
        ``average_attn_over_layers`` is set, the selected per-layer attention view is also
        averaged across layers; otherwise only ``attn_layer`` is used.
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

    def tokens_to_string(self, tokens: List[str]) -> str:
        """Convert a list of decoded tokens to a plain output string."""
        return "".join(tokens)
