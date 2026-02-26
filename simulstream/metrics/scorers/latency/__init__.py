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
    """
    Decorator for registering a latency scorer class.

    Args:
        name (str): The unique identifier for the scorer.

    Raises:
        TypeError: If the decorated class is not a subclass of
            :class:`LatencyScorer`.

    Example:
        >>> @register_latency_scorer("stream_laal")
        ... class StreamLAALScorer(LatencyScorer):
        ...     ...
    """
    def register(cls):
        if not issubclass(cls, LatencyScorer):
            raise TypeError(f"Cannot register {cls.__name__}: must be a subclass of LatencyScorer")
        LATENCY_SCORER_REGISTRY[name] = cls
        return cls

    return register


@dataclass
class LatencyScoringSample:
    """
    Data structure representing a single evaluation sample.

    Attributes:
        audio_name (str): The identifier of the audio file.
        hypothesis (str): The system-generated hypothesis text.
        reference (Optional[List[ReferenceSentenceDefinition]]): One or more reference sentences,
            including the text, start time and duration, or ``None`` if not required.
    """
    audio_name: str
    hypothesis: OutputWithDelays
    reference: Optional[List[ReferenceSentenceDefinition]] = None


@dataclass
class LatencyScores:
    """
    Data structure representing a latency score.

    Attributes:
        ideal_latency (float): The latency score in ideal conditions, which do not include
            computational costs.
        computational_aware_latency (Optional[float]): The latency score in computational aware
            conditions, which include computational costs.
    """
    ideal_latency: float
    computational_aware_latency: Optional[float] = None


class LatencyScorer:
    """
    Abstract base class for all latency scorers.

    A latency scorer evaluates system hypotheses against references and returns a
    :class:`LatencyScores` object that represents the latency scores.

    Subclasses must implement the abstract methods defined here and should be registered via
    :func:`register_latency_scorer`.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """

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


@dataclass
class ResegmentedLatencyScoringSample:
    """
    A sample containing realigned hypotheses and references.

    Attributes:
        audio_name (str): The identifier of the audio file.
        hypothesis (List[str]): Hypothesis lines after realignment.
        reference (List[str]): Reference lines aligned to the hypothesis.
    """
    audio_name: str
    hypothesis: List[OutputWithDelays]
    reference: List[ReferenceSentenceDefinition]


for loader, name, is_pkg in pkgutil.walk_packages(__path__, __name__ + "."):
    importlib.import_module(name)
