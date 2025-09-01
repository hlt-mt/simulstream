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
import importlib
import pkgutil
from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from simulstream.metrics.readers import OutputWithDelays, ReferenceSentenceDefinition


LATENCY_SCORER_REGISTRY = {}


def register_latency_scorer(name):
    def register(cls):
        if not issubclass(cls, LatencyScorer):
            raise TypeError(f"Cannot register {cls.__name__}: must be a subclass of LatencyScorer")
        LATENCY_SCORER_REGISTRY[name] = cls
        return cls

    return register


@dataclass
class LatencyScoringSample:
    audio_name: str
    hypothesis: OutputWithDelays
    reference: Optional[List[ReferenceSentenceDefinition]] = None


@dataclass
class LatencyScores:
    ideal_latency: float
    computational_aware_latency: Optional[float] = None


class LatencyScorer:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    @abstractmethod
    def score(self, samples: List[LatencyScoringSample]) -> LatencyScores:
        ...

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        ...

    @abstractmethod
    def requires_reference(self) -> bool:
        ...


for loader, name, is_pkg in pkgutil.walk_packages(__path__, __name__ + "."):
    importlib.import_module(name)
