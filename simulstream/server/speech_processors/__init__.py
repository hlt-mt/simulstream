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

import importlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List

import numpy as np


CHANNELS = 1
SAMPLE_WIDTH = 2
SAMPLE_RATE = 16_000


@dataclass
class IncrementalOutput:
    """
    Represents the incremental output of a speech processor for a single
    processed chunk of audio.

    Attributes:
        new_tokens (List[str]): List of newly generated tokens in this chunk.
        new_string (str): Concatenated string representation of the new tokens.
        deleted_tokens (List[str]): List of tokens that were deleted/overwritten.
        deleted_string (str): Concatenated string representation of the deleted tokens.
    """
    new_tokens: List[str]
    new_string: str
    deleted_tokens: List[str]
    deleted_string: str

    def strings_to_json(self) -> str:
        """
        Serialize the incremental output to a JSON string.

        Returns:
            str: A JSON string containing the newly generated and the deleted text.
        """
        return json.dumps({"new": self.new_string, "deleted": self.deleted_string})


class SpeechProcessor(ABC):
    """
    Abstract base class for speech processors.

    Subclasses must implement methods to load models, process audio chunks,
    set source/target languages, and clear internal states.
    """

    def __init__(self, config: SimpleNamespace):
        """
        Initialize the speech processor with a given configuration.

        Args:
            config (SimpleNamespace): Configuration loaded from a YAML file.
        """
        self.config = config

    @property
    def speech_chunk_size(self) -> float:
        """
        Return the size of the speech chunks to be processed (in seconds).
        """
        return self.config.speech_chunk_size

    @classmethod
    @abstractmethod
    def load_model(cls, config: SimpleNamespace):
        """
        Load and initialize the underlying speech model.

        Args:
            config (SimpleNamespace): Configuration of the speech processor.
        """
        ...

    @abstractmethod
    def process_chunk(self, waveform: np.float32) -> IncrementalOutput:
        """
        Process a chunk of waveform and produce incremental output.

        Args:
            waveform (np.float32): A 1D NumPy array of the audio chunk. The array is PCM audio
                normalized to the range ``[-1.0, 1.0]`` sampled at
                :attr:`simulstream.server.speech_processors.SAMPLE_RATE`.

        Returns:
            IncrementalOutput: The incremental output (new and deleted tokens/strings).
        """
        ...

    @abstractmethod
    def set_source_language(self, language: str) -> None:
        """
        Set the source language for the speech processor.

        Args:
            language (str): Language code (e.g., ``"en"``, ``"it"``).
        """
        ...

    @abstractmethod
    def set_target_language(self, language: str) -> None:
        """
        Set the target language for the speech processor (for translation).

        Args:
            language (str): Language code (e.g., ``"en"``, ``"it"``).
        """
        ...

    @abstractmethod
    def clear(self) -> None:
        """
        Clear internal states, such as history of cached audio and/or tokens,
        in preparation for a new stream or conversation.
        """
        ...


def build_speech_processor(speech_processor_config: SimpleNamespace) -> SpeechProcessor:
    """
    Instantiate a SpeechProcessor subclass based on configuration.

    The configuration should specify the fully-qualified class name in the
    ``type`` field (e.g. ``"simulstream.server.speech_processors.MyProcessor"``).

    Args:
        speech_processor_config (SimpleNamespace): Configuration for the speech processor.

    Returns:
        SpeechProcessor: An instance of the configured speech processor.

    Raises:
        AssertionError: If the specified class is not a subclass of SpeechProcessor.
    """
    module_path, class_name = speech_processor_config.type.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    assert issubclass(cls, SpeechProcessor), \
        f"{speech_processor_config} must be a subclass of SpeechProcessor"
    cls.load_model(speech_processor_config)
    return cls(speech_processor_config)
