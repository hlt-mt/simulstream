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
from simulstream.server.speech_processors.base_streamatt import (
    BaseStreamAtt,
    FixedWordsTextHistory,
    PunctuationTextHistory,
)

logger = logging.getLogger(__name__)


TEMPLATED_SPEECH_PROMPT = \
    ("You are a professional {src_lang}-to-{tgt_lang} translator. Your goal is to accurately "
     "convey the meaning and nuances of the original {src_lang} speech while adhering to "
     "{tgt_lang} grammar, vocabulary, and cultural sensitivities. Use precise terminology and a "
     "tone appropriate for academic or instructional materials. Produce only the {tgt_lang} "
     "translation, without any additional explanations or commentary. Please translate the "
     "provided {src_lang} speech into {tgt_lang}:")

LANG_MAPPER = {"en": "English", "it": "Italian", "de": "German", "zh": "Chinese (simplified)"}


class DecoderOnlyAttention(BaseStreamAtt):
    """
    Generic Decoder-only Attention-based policy for SpeechLLMs.

    The class handles:
    - Raw-waveform history accumulation.
    - Greedy generation with ``output_attentions=True``.
    - Building the proxy cross-attention matrix from self-attention weights.
    - Applying StreamAtt-based policy on the proxy cross-attention matrix.

    Subclasses must implement the five abstract methods listed below.

    Parameters
    ----------
    config : SimpleNamespace
        All fields from :class:`BaseStreamAtt`, plus:
        attn_layer : int
            Layer from which to extract attention scores. Default: ``0``.
        average_attn_over_layers : bool
            Whether to average attention over all decoder layers instead of
            using the single layer selected by ``attn_layer``.
            Default: ``False``.
        audio_history_max_duration : int
            Maximum raw waveform length to keep in the rolling history.
            Default: ``180`` (seconds).
        max_new_tokens : int
            Maximum tokens to generate per chunk.  Default: ``32``.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.cross_attn_layer = getattr(self.config, "attn_layer", 0)
        self.average_attn_over_layers = getattr(self.config, "average_attn_over_layers", False)
        self.audio_history_max_duration = getattr(self.config, "audio_history_max_duration", 180)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = getattr(self.config, "max_new_tokens", 32)
        self.prefix_summary = ""

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

    def summarize_text(self, prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement summarize_text()."
        )

    def set_target_language(self, language: str) -> None:
        self.tgt_lang = language

    def set_source_language(self, language: str) -> None:
        self.src_lang = language

    def _summary_language_name(self) -> str:
        if getattr(self, "tgt_lang", None):
            return LANG_MAPPER.get(self.tgt_lang, self.tgt_lang)
        if getattr(self, "src_lang", None):
            return LANG_MAPPER.get(self.src_lang, self.src_lang)
        return "the same language as the context"

    def build_raw_text_prefix(self) -> str:
        return "".join(self.text_history) if self.text_history else ""

    def build_summary_context(self) -> str:
        update_prefix_summary = getattr(self.text_history_method, "update_prefix_summary", None)
        if update_prefix_summary is None:
            return ""
        return self.prefix_summary.strip()

    def _update_text_history(self, new_output: List[str]) -> int:
        previous_history = list(self.text_history) if self.text_history else []
        current_history = previous_history + new_output
        discarded_text = super()._update_text_history(new_output)
        update_prefix_summary = getattr(self.text_history_method, "update_prefix_summary", None)
        if discarded_text > 0 and update_prefix_summary is not None:
            new_summary = update_prefix_summary(
                prefix_summary=self.prefix_summary,
                discarded_tokens=current_history[:discarded_text],
                tokens_to_string=self.tokens_to_string,
                summarize_text=self.summarize_text,
                language_name=self._summary_language_name(),
            )
            if new_summary:
                self.prefix_summary = new_summary
        return discarded_text

    def mean_attn_over_heads_and_selected_layers(self, step_attn) -> torch.Tensor:
        if self.average_attn_over_layers:
            return torch.stack(
                [layer_attn[0].mean(dim=0) for layer_attn in step_attn],
                dim=0,
            ).mean(dim=0)
        return step_attn[self.cross_attn_layer][0].mean(dim=0)

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

    def clear(self) -> None:
        super().clear()
        self.prefix_summary = ""


class _SummaryPrefixTextHistory:
    """
    Base summary text-history selector.

    Config attributes
    -----------------
    summary_max_new_tokens : int
        Maximum tokens used when updating the running summary.
    """

    def __init__(self, config: SimpleNamespace, _bow_prefix: str):
        self.summary_max_new_tokens = getattr(config, "summary_max_new_tokens", 32)

    def update_prefix_summary(
            self,
            prefix_summary: str,
            discarded_tokens: List[str],
            tokens_to_string,
            summarize_text,
            language_name: str) -> str:
        discarded_text = tokens_to_string(discarded_tokens).strip()
        if not discarded_text:
            return ""
        if prefix_summary:
            summary_prompt = (
                f"Update these memory notes in {language_name}. Keep them brief and useful for "
                f"continuing the translation. Preserve key entities, terminology, abbreviations, "
                f"numbers, and unresolved references. Return only the updated memory notes.\n\n"
                f"Current memory notes:\n{prefix_summary}\n\n"
                f"New earlier translated context:\n{discarded_text}"
            )
        else:
            summary_prompt = (
                f"Summarize the following earlier translated context in {language_name}. "
                f"Return short memory notes useful for continuing the translation. Preserve key "
                f"entities, terminology, abbreviations, numbers, and unresolved references. "
                f"Prefer concise notes over full prose. Return only the memory notes.\n\n"
                f"Earlier translated context:\n{discarded_text}"
            )
        new_summary = summarize_text(
            summary_prompt,
            self.summary_max_new_tokens,
        ).strip()
        return new_summary


class SummaryFixedWordsTextHistory(FixedWordsTextHistory, _SummaryPrefixTextHistory):
    """
    Fixed-words text-history selector plus a DOA-only running summary prefix.

    Config attributes
    -----------------
    history_words : int
        Number of recent raw words to retain for StreamAtt alignment.
    """

    def __init__(self, config: SimpleNamespace, bow_prefix: str):
        FixedWordsTextHistory.__init__(self, config, bow_prefix)
        _SummaryPrefixTextHistory.__init__(self, config, bow_prefix)


class SummaryPunctuationTextHistory(PunctuationTextHistory, _SummaryPrefixTextHistory):
    """
    Punctuation-based text-history selector plus a DOA-only running summary prefix.

    The raw retained history still follows the punctuation selector, while the
    discarded older context is compressed into a running summary for the next
    decoder-only prompt.
    """

    def __init__(self, config: SimpleNamespace, bow_prefix: str):
        PunctuationTextHistory.__init__(self, config, bow_prefix)
        _SummaryPrefixTextHistory.__init__(self, config, bow_prefix)
