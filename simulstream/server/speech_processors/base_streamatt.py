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

import torch
import logging
import numpy as np

from types import SimpleNamespace
from abc import abstractmethod
from typing import List, Tuple

from simulstream.server.speech_processors import class_load
from simulstream.server.speech_processors.base import BaseSpeechProcessor
from simulstream.server.speech_processors.incremental_output import IncrementalOutput


BOW_PREFIX = "\u2581"


logger = logging.getLogger(__name__)


class BaseStreamAtt(BaseSpeechProcessor):
    """
    A partial implementation of :class:`BaseSpeechProcessor` that provides common logic for the
    StreamAtt policy, introduced in:

       S. Papi, et al. 2024. *"StreamAtt: Direct Streaming Speech-to-Text Translation with
       Attention-based Audio History Selection"* (https://aclanthology.org/2024.acl-long.202/)

    The approach relies on selecting the audio history based on the cross-attention mechanism.
    Specifically, the history for the next decoding step is defined as follows:
     - First, the new textual history is selected by the **text_history_method**, which is in
     charge of selecting the tokens to retain;
     - Second, the new audio history is selected according to cross-attention scores between the
     audio features and the retained textual history by discarding past features that are not
     attended by any tokens of the textual history.

     The derived class should implement the following methods:
        - **audio_max_len**: Returns the maximum audio feature length.
        - **load_model**: Loads the model to device.
        - **_preprocess**: Preprocess the audio features before feeding them into the model.
        - **_generate**: Generate that also returns cross attention scores.

    Args:
       config (SimpleNamespace): Configuration object. The following attributes are expected:
           - **text_history (str)**: config (SimpleNamespace) with the following attribute:
               - **type (str)**: Name of the class to use to determine the text history to keep as
                 context for next predictions.
           - **audio_subsampling_factor (int)**: Subsampling factor of the model, if any.
             Defaults to 1.
           - **text_history_max_len (int)**: The maximum length of the textual history after which
             the current content is cut. Defaults to 128.
           - **cross_attention_layer (int)**: Layer from which to extract the cross-attention from.
           - **cutoff_frame_num (int)**: Number of last frames that cannot be attended by tokens
             in the AlignAtt policy.
           - **word_level_postprocess (bool)**: Whether to postprocess the generated tokens to keep
             only complete words in the emitted hypothesis. To be disabled when operating with
             character-level languages. Defaults to True.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.config = config
        text_history_config = self.config.text_history
        text_history_cls = class_load(text_history_config.type)
        self.text_history_method = text_history_cls(text_history_config)
        self.audio_subsampling_factor = getattr(self.config, "audio_subsampling_factor", 1)
        self.text_history_max_len = getattr(self.config, "text_history_max_len", 128)
        self.cross_attn_layer = getattr(self.config, "cross_attention_layer", 3)
        self.cutoff_frame_num = getattr(self.config, "cutoff_frame_num", 2)
        self.word_level_postprocess = getattr(self.config, "word_level_postprocess", True)
        self.unselected_tokens = []

    @property
    @abstractmethod
    def audio_max_len(self) -> float:
        """
        Return the maximum allowed length of the audio features, beyond which the audio is cut off.
        """
        ...

    @abstractmethod
    def _generate(self, speech: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        """
        Generate tokens from the given speech features together with the cross-attention scores.

        Args:
            speech (torch.Tensor): Model-ready speech features as produced by :meth:`_preprocess`.

        Returns:
            Tuple[List[str], torch.Tensor]:
                List[str]: A list of generated tokens.
                torch.Tensor: Cross-attention scores with dimension (generated_tokens,
                input_length).
        """
        ...

    @staticmethod
    def normalize_attn(attn):
        """
        Normalize the cross-attention scores along the frame dimension to avoid attention sinks.
        """
        std = attn.std(axis=0)
        std[std == 0.] = 1.0
        mean = attn.mean(axis=0)
        return (attn - mean) / std

    def _update_text_history(self, new_output: List[str]) -> int:
        if self.text_history:
            self.text_history += new_output
        else:
            self.text_history = new_output
        new_history = self.text_history_method.select_text_history(self.text_history)
        discarded_text = len(self.text_history) - len(new_history)
        self.text_history = new_history

        # Ensure not exceeding max text history length
        if self.text_history and len(self.text_history) > self.text_history_max_len:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    f"The textual history has hit the maximum predefined length of "
                    f"{self.text_history_max_len}")
            self.text_history = self.text_history[-self.text_history_max_len:]
        return discarded_text

    def _cut_audio_exceeding_maxlen(self):
        # Ensure not exceeding max audio history length
        if len(self.audio_history) > self.audio_max_len:
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    f"The audio history has hit the maximum predefined length of "
                    f"{self.audio_max_len}")
            self.audio_history = self.audio_history[-self.audio_max_len:]

    def _update_speech_history(self, discarded_text: int, cross_attn: torch.Tensor) -> None:
        # If no history is discarded, no need for attention-based audio trimming
        if discarded_text == 0:
            # Check audio history not exceeding maximum allowed length
            self._cut_audio_exceeding_maxlen()
            return

        assert len(self.text_history) > 0, \
            "If text history is empty after selection, audio cannot be aligned. " \
            "If you see this message, it indicates a bug, so please open an issue at " \
            "https://github.com/hlt-mt/simulstream/issues and include the steps that " \
            "led to this state."

        # Trim the cross-attention by excluding the discarded new generated tokens and the
        # discarded textual history. Output shape: (text_history_len, n_audio_features)
        cross_attn = cross_attn[discarded_text:discarded_text + len(self.text_history), :]

        # Compute the frame to which each token of the textual history mostly attends to
        most_attended_idxs = torch.argmax(cross_attn.float(), dim=1)

        # Find the first feature that is attended
        if most_attended_idxs.shape[0] > 1:
            # Multiple tokens: sort and get the earliest attended frame
            sorted_idxs = torch.sort(most_attended_idxs)[0]
            earliest_attended_idx = sorted_idxs[0]
        else:
            # Only one token: use the unique most attended frame
            earliest_attended_idx = most_attended_idxs[0]

        # Multiply by the subsampling factor to recover the original number of frames
        frames_to_cut = earliest_attended_idx * self.audio_subsampling_factor

        # Cut the unattended audio features
        self.audio_history = self.audio_history[frames_to_cut:]

        # Check audio history not exceeding maximum allowed length
        self._cut_audio_exceeding_maxlen()

    @staticmethod
    def _strip_incomplete_words(tokens: List[str]) -> List[str]:
        """
        Remove last incomplete word(s) from the new hypothesis.

        Args:
            tokens (List[str]): selected tokens, possibly containing partial words to be removed.

        Returns:
            List[str]: A list of generated tokens from which partial words are removed.
        """
        tokens_to_write = []
        # iterate from the end and count how many trailing tokens to drop
        num_tokens_incomplete = 0
        for tok in reversed(tokens):
            num_tokens_incomplete += 1
            if tok.startswith(BOW_PREFIX):
                # slice off the trailing incomplete tokens
                tokens_to_write = tokens[:-num_tokens_incomplete]
                break
        return tokens_to_write

    def alignatt_policy(self, generated_tokens, cross_attn) -> List[str]:
        """
        Apply the AlignAtt policy by cutting off tokens whose attention falls
        beyond the allowed frame range.
        The AlignAtt policy was introduced in:
            S. Papi, et al. 2023. *"AlignAtt: Using Attention-based Audio-Translation
            Alignments as a Guide for Simultaneous Speech Translation"*
            (https://www.isca-archive.org/interspeech_2023/papi23_interspeech.html)
        """
        # Select attention scores corresponding to the new generated tokens
        cross_attn = cross_attn[-len(generated_tokens):, :]
        selected_tokens = generated_tokens

        # Find the frame to which each token mostly attends to
        most_attended_frames = torch.argmax(cross_attn, dim=1)
        cutoff = cross_attn.size(1) - self.cutoff_frame_num

        # Find the first token that attends beyond the cutoff frame
        invalid_tok_ids = torch.where(most_attended_frames >= cutoff)[0]

        # Truncate tokens up to the first invalid alignment (if any)
        if len(invalid_tok_ids) > 0:
            selected_tokens = selected_tokens[:invalid_tok_ids[0]]

        if self.word_level_postprocess:
            selected_tokens = self._strip_incomplete_words(selected_tokens)

        # Store unselected tokens, to be used in the case of end of stream
        self.unselected_tokens = generated_tokens[len(selected_tokens):]

        return selected_tokens

    def _build_incremental_outputs(self, generated_tokens: List[str]) -> IncrementalOutput:
        return IncrementalOutput(
            new_tokens=generated_tokens,
            new_string=self.tokens_to_string(generated_tokens),
            deleted_tokens=[],
            deleted_string="",
        )

    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        speech = self._preprocess(waveform)
        # Generate new hypothesis with its corresponding cross-attention scores (no prefix)
        generated_tokens, cross_attn = self._generate(speech)
        # Select the part of the new hypothesis to be emitted, and trim cross-attention accordingly
        selected_output = self.alignatt_policy(generated_tokens, cross_attn)
        incremental_output = self._build_incremental_outputs(selected_output)
        # Discard textual history, if needed
        discarded_text = self._update_text_history(selected_output)
        # Trim audio corresponding to the discarded textual history
        self._update_speech_history(discarded_text, cross_attn)
        return incremental_output

    def end_of_stream(self) -> IncrementalOutput:
        last_output = self._build_incremental_outputs(self.unselected_tokens)
        self.unselected_tokens = []
        return last_output

    def clear(self) -> None:
        super().clear()
        self.text_history = None
        self.audio_history = None
        self.unselected_tokens = []


class FixedWordsTextHistory:
    """
    Fixed Words textual history selection method that retains a pre-defined
    number of words in the history (*history_words*).

    The current implementation supports only SentencePiece.
    """
    def __init__(self, config: SimpleNamespace):
        self.history_words = getattr(config, "history_words", 20)
        self.config = config

    def select_text_history(self, text_history: List[str]):
        words_to_keep = self.history_words
        new_history = []
        for token in reversed(text_history):
            new_history.append(token)
            # Check if 'BOW_PREFIX' (space in SentencePiece) is contained in the token,
            # meaning that we reached the beginning of the word that should be counted
            if BOW_PREFIX in token:
                words_to_keep -= 1
                # When all the words to keep are consumed, the accumulation is stopped
                # and the prefix is returned
                if words_to_keep == 0:
                    break
        # Reverse the list
        return new_history[::-1]


class PunctuationTextHistory:
    """
    Punctuation textual history selection method that retains the sentence
    before the last strong punctuation character.

    The current implementation supports only SentencePiece.
    """

    STRONG_PUNCTUATION = [".", "!", "?", ":", ";", "。"]

    def __init__(self, config: SimpleNamespace):
        self.config = config

    def select_text_history(self, text_history):
        new_history = []
        seen_punctuation = False

        for token in reversed(text_history):
            prefix_token = token
            contains_punctuation = False
            for punct in self.STRONG_PUNCTUATION:
                if punct in prefix_token:
                    contains_punctuation = True
                    break
            if contains_punctuation:
                if seen_punctuation:
                    break
            seen_punctuation = True
            new_history.append(token)
        # Reverse the list
        return new_history[::-1]
