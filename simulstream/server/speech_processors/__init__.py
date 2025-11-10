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
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import List, Any

import numpy as np

from simulstream.server.speech_processors.incremental_output import IncrementalOutput


CHANNELS = 1
SAMPLE_WIDTH = 2
SAMPLE_RATE = 16_000


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
    def end_of_stream(self) -> IncrementalOutput:
        """
        This method is called at the end of audio chunk processing. It can be used to emit
        hypotheses at the end of the speech to conclude the output.

        Returns:
            IncrementalOutput: The incremental output (new and deleted tokens/strings).
        """
        ...

    @abstractmethod
    def tokens_to_string(self, tokens: List[str]) -> str:
        """
        Converts token sequences into human-readable strings.

        Returns:
            str: The textual representation of the tokens.
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
    cls = speech_processor_class_load(speech_processor_config.type)
    cls.load_model(speech_processor_config)
    return cls(speech_processor_config)


def speech_processor_class_load(speech_processor_class_string: str) -> type[SpeechProcessor]:
    """
    Import the speech processor class from its string definition.

    Args:
        speech_processor_class_string (str): Full name of the speech processor class to load.

    Returns:
        SpeechProcessorClass: A class object for the speech processor class.

    Raises:
        AssertionError: If the specified class is not a subclass of SpeechProcessor.
    """
    cls = class_load(speech_processor_class_string)
    assert issubclass(cls, SpeechProcessor), \
        f"{speech_processor_class_string} must be a subclass of SpeechProcessor"
    return cls


def class_load(class_string: str) -> type[Any]:
    module_path, class_name = class_string.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
