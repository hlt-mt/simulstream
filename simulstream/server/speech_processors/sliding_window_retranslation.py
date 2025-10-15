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

from abc import abstractmethod
from difflib import SequenceMatcher
from types import SimpleNamespace
from typing import List

import torch

from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.base import BaseSpeechProcessor, IncrementalOutput


class SlidingWindowRetranslator(BaseSpeechProcessor):
    """
    A speech processor that applies a fixed-length sliding window retranslation with
    deduplication to mitigate overlapping outputs when processing unsegmented audio streams.

    This class implements the algorithm introduced in:

       S. Sen, et al. 2025. *"Simultaneous Translation for Unsegmented Input:
       A Sliding Window Approach"* (https://arxiv.org/pdf/2210.09754)

    The approach relies on detecting the **longest common subsequence** between the current window
    and the previous one, in order to prevent repeating tokens caused by overlapping audio windows.

    Subclasses must implement :meth:`_tokens_to_string` to define how token sequences are converted
    into human-readable strings.

    Args:
       config (SimpleNamespace): Configuration object. The following attributes are expected:

           - **window_len (int)**: Length of the sliding window (in seconds).
           - **matching_threshold (float, optional)**: Minimum fraction of the current tokens that
             must match the previous history to be considered aligned. Default = ``0.1``.
           - **override_on_failed_match (bool, optional)**: If ``True``, the previous history is
             deleted from the output when no sufficient match is found. Otherwise, previous history
             is kept and the new output is appended to the end of the previous history.
             Default = ``False``.
           - **max_tokens_per_second (int, optional)**: Maximum output tokens allowed per second of
             audio. Default = ``10``.
    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.window_len = self.config.window_len * SAMPLE_RATE
        self.matching_threshold = getattr(self.config, "matching_threshold", 0.1)
        self.override_on_failed_match = getattr(self.config, "override_on_failed_match", False)
        self.max_tokens_per_second = getattr(self.config, "max_tokens_per_second", 10)
        self.within_first_window = True

    @abstractmethod
    def _tokens_to_string(self, tokens: List[str]) -> str:
        ...

    def _build_incremental_outputs(self, generated_tokens: List[str]) -> IncrementalOutput:
        """
        Deduplicates the output stream of overlapping windows using the algorithm introduced in
        `S. Sen, et al. 2025. "Simultaneous Translation for Unsegmented Input:
        A Sliding Window Approach" <https://arxiv.org/pdf/2210.09754>`_

        This algorithm is based on the longest matching substring between the current and previous
        window. We use tokens instead of string to match, though, as we have empirically observed
        that tokenization is mostly consistent across generations of the same word.
        """
        if self.text_history is None or len(self.text_history) == 0:
            self.text_history = generated_tokens
            generated_string = self._tokens_to_string(generated_tokens)
            return IncrementalOutput(
                new_tokens=generated_tokens,
                new_string=generated_string,
                deleted_tokens=[],
                deleted_string=""
            )
        seq_matcher = SequenceMatcher(None, self.text_history, generated_tokens, autojunk=False)
        longest_match = seq_matcher.find_longest_match()
        if longest_match.size >= self.matching_threshold * len(generated_tokens):
            new_tokens = generated_tokens[longest_match.b + longest_match.size:]
            deleted_tokens = self.text_history[longest_match.a + longest_match.size:]
            new_string = self._tokens_to_string(new_tokens)
            deleted_string = self._tokens_to_string(deleted_tokens)
            # we take the matching part and the last part of the generated string as part of
            # the history. Then we take from the history the tokens corresponding to the amount
            # generated in this step, to ensure we have a sufficiently wide window
            matching_and_last_tokens = generated_tokens[longest_match.b:]
            initial_discarded_tokens = len(generated_tokens) - len(matching_and_last_tokens)
            history_tokens_discarded = self.text_history[longest_match.a:]
            history_initial_tokens = len(self.text_history) - len(history_tokens_discarded)
            new_history_initial_tokens = self.text_history[
                max(history_initial_tokens - initial_discarded_tokens, 0):history_initial_tokens]
            self.text_history = new_history_initial_tokens + matching_and_last_tokens
        else:
            if self.within_first_window or self.override_on_failed_match:
                deleted_tokens = self.text_history
                deleted_string = self._tokens_to_string(self.text_history)
            else:
                deleted_tokens = []
                deleted_string = ""
            new_tokens = generated_tokens
            new_string = self._tokens_to_string(generated_tokens)
            self.text_history = generated_tokens
        return IncrementalOutput(
            new_tokens=new_tokens,
            new_string=new_string,
            deleted_tokens=deleted_tokens,
            deleted_string=deleted_string,
        )

    def _update_speech_history(
            self,
            new_speech: torch.Tensor,
            generated_tokens: List[str],
            new_output: IncrementalOutput) -> None:
        if self.within_first_window and len(self.audio_history) >= self.window_len:
            self.within_first_window = False

    def _update_text_history(
            self,
            new_speech: torch.Tensor,
            generated_tokens: List[str],
            new_output: IncrementalOutput) -> None:
        pass

    def clear(self) -> None:
        super().clear()
        self.within_first_window = True
