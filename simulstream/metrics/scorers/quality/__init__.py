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


QUALITY_SCORER_REGISTRY = {}


def register_quality_scorer(name):
    def register(cls):
        if not issubclass(cls, QualityScorer):
            raise TypeError(f"Cannot register {cls.__name__}: must be a subclass of QualityScorer")
        QUALITY_SCORER_REGISTRY[name] = cls
        return cls

    return register


@dataclass
class QualityScoringSample:
    audio_name: str
    hypothesis: str
    reference: Optional[List[str]] = None
    source: Optional[List[str]] = None


class QualityScorer:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    @abstractmethod
    def score(self, samples: List[QualityScoringSample]) -> float:
        ...

    @classmethod
    @abstractmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        ...

    @abstractmethod
    def requires_source(self) -> bool:
        ...

    @abstractmethod
    def requires_reference(self) -> bool:
        ...


for loader, name, is_pkg in pkgutil.walk_packages(__path__, __name__ + "."):
    importlib.import_module(name)
