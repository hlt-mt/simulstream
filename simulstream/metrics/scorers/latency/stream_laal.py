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

import argparse
import logging
import statistics
from typing import List

from simulstream.metrics.readers import text_items
from simulstream.metrics.scorers.latency import register_latency_scorer, LatencyScores
from simulstream.metrics.scorers.latency.mwersegmenter import MWERSegmenterBasedLatencyScorer, \
    ResegmentedLatencyScoringSample


LOGGER = logging.getLogger('simulstream.metrics.scorers.latency.stream_laal')


@register_latency_scorer("stream_laal")
class StreamLaal(MWERSegmenterBasedLatencyScorer):
    """
    Computes StreamLAAL version 2.0, as proposed in
    `StreamAtt: Direct Streaming Speech-to-Text Translation with Attention-based
    Audio History Selection <https://aclanthology.org/2024.acl-long.202.pdf>`_.


    Then main difference with version 1 is the different segmentation of
    the text (uses mwerSegmenter python package instead of Matusov's executable).
    """

    @staticmethod
    def _sentence_level_laal(
            delays: List[float], source_length: float, target_length: int) -> float:
        """
        Function to compute Length Adaptive Average Lagging (LAAL) on one sentence as proposed in
        `CUNI-KIT System for Simultaneous Speech Translation Task at IWSLT 2022
        <https://arxiv.org/abs/2204.06028>`_ and
        `Length-Adaptive Average Lagging for Simultaneous Speech Translation
        <https://arxiv.org/abs/2206.05807>`_.
        It is the original Average Lagging as proposed in
        `Controllable Latency using Prefix-to-Prefix Framework
        <https://arxiv.org/abs/1810.08398>`_
        but is robust to the length difference between the hypothesis and reference.

        The implementation is derived by that available in SimulEval (see `latency_scorer.py` in
        `https://github.com/facebookresearch/SimulEval/).

        Returns:
            float: the latency score on one sentence.
        """
        if delays[0] > source_length:
            return delays[0]

        LAAL = 0
        gamma = max(len(delays), target_length) / source_length
        tau = 0
        for t_minus_1, d in enumerate(delays):
            LAAL += d - t_minus_1 / gamma
            tau = t_minus_1 + 1

            if d >= source_length:
                break
        LAAL /= tau
        return LAAL

    def _do_score(self, samples: List[ResegmentedLatencyScoringSample]) -> LatencyScores:
        sentence_level_ideal_scores = []
        sentence_level_ca_scores = []
        skipped_sentences = 0
        for sample in samples:
            for sentence_output, sentence_reference in zip(sample.hypothesis, sample.reference):
                # offset delays with respect to reference start of the utterance
                delays_from_sentence_start = [
                    delay - sentence_reference.start_time
                    for delay in sentence_output.ideal_delays]
                ca_delays_from_sentence_start = [
                    delay - sentence_reference.start_time
                    for delay in sentence_output.computational_aware_delays]
                assert len(delays_from_sentence_start) == len(ca_delays_from_sentence_start)

                target_length = len(text_items(sentence_reference.content, self.latency_unit))

                if len(delays_from_sentence_start) > 0:
                    sentence_level_ideal_scores.append(
                        self._sentence_level_laal(
                            delays_from_sentence_start,
                            sentence_reference.duration,
                            target_length)
                    )
                    sentence_level_ca_scores.append(
                        self._sentence_level_laal(
                            ca_delays_from_sentence_start,
                            sentence_reference.duration,
                            target_length)
                    )
                else:
                    skipped_sentences += 1

        if skipped_sentences > 0:
            LOGGER.warning(
                f"{skipped_sentences} sentences have been skipped in LAAL computation as they "
                "were empty")
        return LatencyScores(
            statistics.mean(sentence_level_ideal_scores),
            statistics.mean(sentence_level_ca_scores))

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        pass
