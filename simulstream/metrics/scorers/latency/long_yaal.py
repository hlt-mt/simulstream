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

import argparse
import logging
import statistics
from typing import List, Optional

from simulstream.metrics.readers import text_items
from simulstream.metrics.scorers.latency import register_latency_scorer, LatencyScores
from simulstream.metrics.scorers.latency.softsegmenter import (
    SoftSegmenterBasedLatencyScorer,
    ResegmentedLatencyScoringSample
)


LOGGER = logging.getLogger('simulstream.metrics.scorers.latency.long_yaal')


@register_latency_scorer("long_yaal")
class LongYAAL(SoftSegmenterBasedLatencyScorer):
    """
    Computes Long-form Yet Another Average Lagging (LongYAAL) as proposed in
    `Better Late Than Never: Evaluation of Latency Metrics for Simultaneous Speech-to-Text
    Translation <https://arxiv.org/abs/2509.17349>`_.

    This metric uses SoftSegmenter alignment to realign system outputs to reference segments
    before computing latency, making it more robust for long-form speech translation evaluation.

    The key difference from StreamLAAL is the use of SoftSegmenter's more sophisticated
    alignment algorithm that handles long-form audio better. Additionally, LongYAAL is considers
    all output tokens up until the end of the recording. StreamLAAL ignores any output tokens 
    emitted after the end of the reference segments.
    """

    @staticmethod
    def _sentence_level_yaal(
        delays: List[float],
        source_length: float,
        target_length: int,
        recording_end: float = float("inf"),
        is_longform: bool = True,
    ) -> Optional[float]:
        """
        Compute Yet Another Average Lagging (YAAL) on one sentence.

        Args:
            delays (List[float]): Sequence of delays for each output token.
            source_length (float): Length of the source audio segment in milliseconds.
            target_length (int): Length of the target reference in tokens/characters.
            recording_end (float): End time of the recording (for long-form audio).
            is_longform (bool): Whether to treat as long-form audio (allows delays past
            source_length).

        Returns:
            Optional[float]: The YAAL score for the sentence, or None if computation is
            not possible.
        """
        if len(delays) == 0:
            return None

        # If the first delay is already past the end, we can't compute YAAL
        if (delays[0] >= source_length and not is_longform) or (
            delays[0] >= recording_end
        ):
            return None

        YAAL = 0.0
        gamma = max(len(delays), target_length) / source_length
        tau = 0

        for t_minus_1, d in enumerate(delays):
            # Stop if we've exceeded the source length (for non-longform)
            # or recording end (for longform)
            if (d >= source_length and not is_longform) or (d >= recording_end):
                break

            YAAL += d - t_minus_1 / gamma
            tau = t_minus_1 + 1

        if tau == 0:
            return None

        YAAL /= tau
        return YAAL

    def _do_score(
        self, samples: List[ResegmentedLatencyScoringSample]
    ) -> LatencyScores:
        sentence_level_ideal_scores = []
        sentence_level_ca_scores = []
        skipped_sentences = 0

        for sample in samples:
            # Compute the total recording length (end time of the last reference segment)
            if sample.reference:
                recording_length = max(
                    ref.start_time + ref.duration for ref in sample.reference
                )
            else:
                LOGGER.warning(
                    f"Sample {sample.audio_name} has no reference segments; treating recording"
                    " length as infinite"
                )
                recording_length = float("inf")

            for sentence_output, sentence_reference in zip(
                sample.hypothesis, sample.reference
            ):
                # Note: delays in sentence_output are already offset relative to
                # sentence_reference.start_time
                # by the SoftSegmenter alignment (unlike MWERSegmenter which doesn't offset)
                ideal_delays = sentence_output.ideal_delays
                ca_delays = sentence_output.computational_aware_delays

                assert len(ideal_delays) == len(
                    ca_delays
                ), f"Mismatch in delay counts: {len(ideal_delays)} vs {len(ca_delays)}"

                target_length = len(
                    text_items(sentence_reference.content, self.latency_unit)
                )

                if len(ideal_delays) > 0:
                    # Compute recording end time relative to sentence start
                    # This considers the entire recording, not just this segment.
                    # This allows LongYAAL to account for outputs emitted after the reference
                    # segment ends but before the recording ends (key difference from StreamLAAL)
                    recording_end = recording_length - sentence_reference.start_time

                    ideal_score = self._sentence_level_yaal(
                        ideal_delays,
                        sentence_reference.duration,
                        target_length,
                        recording_end=recording_end,
                        is_longform=True,
                    )

                    ca_score = self._sentence_level_yaal(
                        ca_delays,
                        sentence_reference.duration,
                        target_length,
                        recording_end=recording_end,
                        is_longform=True,
                    )

                    if ideal_score is not None:
                        sentence_level_ideal_scores.append(ideal_score)
                    else:
                        skipped_sentences += 1
                    if ca_score is not None:
                        sentence_level_ca_scores.append(ca_score)
                else:
                    skipped_sentences += 1

        if skipped_sentences > 0:
            LOGGER.warning(
                f"{skipped_sentences} sentences have been skipped in LongYAAL computation "
                f"as they were empty or could not be scored"
            )

        if len(sentence_level_ideal_scores) == 0:
            LOGGER.error("No sentences could be scored for LongYAAL")
            return LatencyScores(float("nan"), float("nan"))

        return LatencyScores(
            statistics.mean(sentence_level_ideal_scores),
            (
                statistics.mean(sentence_level_ca_scores)
                if len(sentence_level_ca_scores) > 0
                else float("nan")
            ),
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--moses-lang",
            type=str,
            default=None,
            help='Language code for Moses tokenizer (e.g., "en", "de"). '
            "Use None for Chinese/Japanese or to skip tokenization.",
        )
