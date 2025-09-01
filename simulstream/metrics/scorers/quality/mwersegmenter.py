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
from typing import List, Optional

from mweralign import mweralign

from simulstream.metrics.scorers.quality import QualityScorer, QualityScoringSample


@dataclass
class ResegmentedQualityScoringSample:
    audio_name: str
    hypothesis: List[str]
    reference: List[str]
    source: Optional[List[str]] = None


class MWERSegmenterBasedQualityScorer(QualityScorer):
    def requires_reference(self) -> bool:
        return True

    @abstractmethod
    def _do_score(self, samples: List[ResegmentedQualityScoringSample]) -> float:
        ...

    def score(self, samples: List[QualityScoringSample]) -> float:
        resegmented_samples = []
        for sample in samples:
            assert sample.reference is not None, "Cannot realign hypothesis to missing reference"
            resegmented_hypos = mweralign.align_texts(
                "\n".join(sample.reference), sample.hypothesis).split("\n")
            assert len(sample.reference) == len(resegmented_hypos), \
                f"Reference ({sample.audio_name}) has mismatched number of target " \
                f"({len(sample.reference)}) and resegmented lines ({len(resegmented_hypos)})"
            resegmented_samples.append(ResegmentedQualityScoringSample(
                sample.audio_name,
                resegmented_hypos,
                sample.reference,
                sample.source
            ))
        return self._do_score(resegmented_samples)
