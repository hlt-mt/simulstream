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
from dataclasses import dataclass
from typing import List

from mweralign import mweralign

from simulstream.metrics.readers import ReferenceSentenceDefinition, OutputWithDelays, text_items
from simulstream.metrics.scorers.latency import LatencyScorer, LatencyScoringSample, LatencyScores


@dataclass
class ResegmentedLatencyScoringSample:
    audio_name: str
    hypothesis: List[OutputWithDelays]
    reference: List[ReferenceSentenceDefinition]


class MWERSegmenterBasedLatencyScorer(LatencyScorer):
    def __init__(self, args):
        super().__init__(args)
        self.latency_unit = args.latency_unit

    def requires_reference(self) -> bool:
        return True

    @abstractmethod
    def _do_score(self, samples: List[ResegmentedLatencyScoringSample]) -> LatencyScores:
        ...

    def _split_delays_by_segmented_text(
            self, delays: List[float], segmented_text: List[str]) -> List[List[float]]:
        segmented_delays = []
        index = 0

        for segment in segmented_text:
            segment_len = len(text_items(segment, self.latency_unit))
            segmented_delays.append(delays[index:index + segment_len])
            index += segment_len
        assert len(delays) == index, f"Index {index} should have reached end of delays ({len(delays)})"
        return segmented_delays

    def score(self, samples: List[LatencyScoringSample]) -> LatencyScores:
        resegmented_samples = []
        for sample in samples:
            assert sample.reference is not None, "Cannot realign hypothesis to missing reference"

            resegmented_hypos = mweralign.align_texts(
                "\n".join([sentence_def.content for sentence_def in sample.reference]),
                sample.hypothesis.final_text).split("\n")

            assert len(resegmented_hypos) == len(sample.reference), \
                f"Reference ({sample.audio_name}) has mismatched number of target " \
                f"({len(sample.reference)}) and resegmented lines ({len(resegmented_hypos)})"

            ideal_delays_splits = self._split_delays_by_segmented_text(
                sample.hypothesis.ideal_delays,
                resegmented_hypos)
            computational_aware_delays_splits = self._split_delays_by_segmented_text(
                sample.hypothesis.computational_aware_delays,
                resegmented_hypos)
            assert len(ideal_delays_splits) == len(computational_aware_delays_splits)

            resegmented_hypos_with_delays = []
            for text, ideal_delay, computational_aware_delay in zip(
                    resegmented_hypos, ideal_delays_splits, computational_aware_delays_splits):
                resegmented_hypos_with_delays.append(
                    OutputWithDelays(text, ideal_delay, computational_aware_delay))

            resegmented_samples.append(ResegmentedLatencyScoringSample(
                sample.audio_name,
                resegmented_hypos_with_delays,
                sample.reference,
            ))
        return self._do_score(resegmented_samples)
