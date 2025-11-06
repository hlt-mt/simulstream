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

import json
import logging
import time
from typing import Optional

import librosa
import numpy as np

from simulstream.metrics.logger import METRICS_LOGGER
from simulstream.server.speech_processors import SpeechProcessor, SAMPLE_RATE
from simulstream.server.speech_processors.incremental_output import merge_incremental_outputs, \
    IncrementalOutput


LOGGER = logging.getLogger('simulstream.message_processor')


class MessageProcessor:
    """
    This class is responsible for processing the messages incoming from a client, which include
    control messages (e.g., configurations about languages to use, or signal that the stream is
    over).
    """
    def __init__(self, client_id: int, speech_processor: SpeechProcessor):
        self.client_buffer = b''
        self.processed_audio_seconds = 0
        self.sample_rate = SAMPLE_RATE
        self.client_id = client_id
        self.speech_processor = speech_processor

    def process_speech(self, speech_data: bytes) -> Optional[IncrementalOutput]:
        """
        Process an audio chunk and return incremental transcription/translation.
        Namely, it receives and buffers raw audio chunks (``bytes``) and processes audio
        audio incrementally with the configured
        :class:`~simulstream.server.speech_processors.SpeechProcessor`.

        Args:
            speech_data (bytes): Raw PCM audio bytes (16-bit little endian).

        Returns:
            IncrementalOutput: incremental processing results, if any. None otherwise.
        """
        self.client_buffer += speech_data
        # we have SAMPLE_RATE * 2 bytes (int16) samples every second
        buffer_len_seconds = len(self.client_buffer) / 2 / self.sample_rate
        if buffer_len_seconds >= self.speech_processor.speech_chunk_size:
            self.processed_audio_seconds += buffer_len_seconds
            start_time = time.time()
            incremental_output = self._run_speech_processor()
            end_time = time.time()
            METRICS_LOGGER.info(json.dumps({
                "id": self.client_id,
                "total_audio_processed": self.processed_audio_seconds,
                "computation_time": end_time - start_time,
                "generated_tokens": incremental_output.new_tokens,
                "deleted_tokens": incremental_output.deleted_tokens,
            }))
            return incremental_output
        else:
            return None

    def _run_speech_processor(self) -> IncrementalOutput:
        """
        This function converts raw ``int16`` PCM audio to normalized ``float32``,
        resamples it if necessary to :data:`~simulstream.server.speech_processors.SAMPLE_RATE`,
        and forwards it to the given class:`~simulstream.server.speech_processors.SpeechProcessor`.
        Processing statistics are logged using the metrics logger.
        """
        int16_waveform = np.frombuffer(self.client_buffer, dtype=np.int16)
        float32_waveform = int16_waveform.astype(np.float32) / 2 ** 15
        if self.sample_rate != SAMPLE_RATE:
            float32_waveform = librosa.resample(
                float32_waveform, orig_sr=self.sample_rate, target_sr=SAMPLE_RATE)
        incremental_output = self.speech_processor.process_chunk(float32_waveform)
        self.client_buffer = b''
        return incremental_output

    def process_metadata(self, metadata: dict):
        """
        Takes a dictionary of metadata regarding the incoming speech and desired output, and
        interacts with the configured speech_processor to set it up accordingly.

        Args:
            metadata (dict): Dictionary of metadata regarding the incoming speech.
        """
        if 'sample_rate' in metadata:
            self.sample_rate = int(metadata['sample_rate'])
        if 'target_lang' in metadata:
            self.speech_processor.set_target_language(metadata["target_lang"])
            LOGGER.debug(
                f"Client {self.client_id} target language set to: "
                f"{metadata['target_lang']}")
        if 'source_lang' in metadata:
            self.speech_processor.set_source_language(metadata["source_lang"])
            LOGGER.debug(
                f"Client {self.client_id} source language set to: {metadata['source_lang']}")
        if 'metrics_metadata' in metadata:
            METRICS_LOGGER.info(json.dumps({
                "id": self.client_id,
                "metadata": metadata["metrics_metadata"]
            }))
            LOGGER.debug(
                f"Logged client {self.client_id} metrics metadata: {metadata['metrics_metadata']}")

    def end_of_stream(self) -> IncrementalOutput:
        """
        Performs the last operations to conclude the processing of the stream of audio by the
        speech processor and cleans up everything to be ready for the next stream.

        Returns:
            IncrementalOutput: last output at the end of the stream.
        """
        outputs = []
        start_time = time.time()
        if self.client_buffer:
            # process remaining audio after last chunk
            self.processed_audio_seconds += len(self.client_buffer) / 2 / self.sample_rate
            outputs.append(self._run_speech_processor())

        outputs.append(self.speech_processor.end_of_stream())
        incremental_output = merge_incremental_outputs(
            outputs, self.speech_processor.tokens_to_string)
        end_time = time.time()
        METRICS_LOGGER.info(json.dumps({
            "id": self.client_id,
            "total_audio_processed": self.processed_audio_seconds,
            "computation_time": end_time - start_time,
            "generated_tokens": incremental_output.new_tokens,
            "deleted_tokens": incremental_output.deleted_tokens,
        }))
        self.clear()
        return incremental_output

    def clear(self):
        """
        Clear the internal states to be ready for a new input stream.
        """
        self.speech_processor.clear()
        self.client_buffer = b''
        self.processed_audio_seconds = 0
        self.sample_rate = SAMPLE_RATE
