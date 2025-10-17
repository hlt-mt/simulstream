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
import json
import logging
import time
from types import SimpleNamespace
from typing import List, Optional

import numpy as np

import simulstream
from simulstream.client.wav_reader_client import load_wav_file_list, read_wav_file
from simulstream.config import yaml_config
from simulstream.metrics.logger import setup_metrics_logger, METRICS_LOGGER
from simulstream.server.message_processor import MessageProcessor
from simulstream.server.speech_processors import build_speech_processor, SpeechProcessor


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGER = logging.getLogger('simulstream.inference')


def process_audio(
        message_processor: MessageProcessor,
        sample_rate: int,
        data: np.ndarray):
    """
    Stream audio data in fixed-size chunks over a WebSocket connection.

    Args:
        message_processor (MessageProcessor): class that processes the audio chunks.
        sample_rate (int): Audio sample rate (Hz).
        data (np.ndarray): Audio samples as int16 array.
    """
    samples_per_chunk = int(
        sample_rate * message_processor.speech_processor.speech_chunk_size / 1000.0)
    i = 0
    for i in range(0, len(data), samples_per_chunk):
        output = message_processor.process_speech(data[i:i + samples_per_chunk].tobytes())
        LOGGER.debug(f"response: {output}")
    # send last part of the audio
    if i < len(data):
        output = message_processor.process_speech(data[i:].tobytes())
        LOGGER.debug(f"response: {output}")


def run_inference(
        speech_processor: SpeechProcessor,
        wav_file_list: List[str],
        tgt_lang: Optional[str] = None,
        src_lang: Optional[str] = None):
    """
    Runs the inference on the WAV files sequentially with the specified speech processor.

    For each file:
      - Sets metadata (sample rate, filename, optional languages).
      - Processes the audio in chunks.

    Args:
        speech_processor (SpeechProcessor): the speech processor to use to run the inference.
        wav_file_list (list[str]): Paths to WAV files.
        tgt_lang (str | None): Target language code (e.g., "en").
        src_lang (str | None): Source language code (e.g., "en").
    """
    for i, wav_file in enumerate(wav_file_list):
        LOGGER.info(f"Streaming: {wav_file}")
        sample_rate, data = read_wav_file(wav_file)
        metadata = {
            "sample_rate": sample_rate,
            "metrics_metadata": {
                "wav_name": wav_file,
            }
        }
        if tgt_lang is not None:
            metadata["target_lang"] = tgt_lang
        if src_lang is not None:
            metadata["source_lang"] = src_lang
        message_processor = MessageProcessor(i, speech_processor)
        message_processor.process_metadata(metadata)
        process_audio(message_processor, sample_rate, data)
        message_processor.end_of_stream()
    LOGGER.info(f"All {len(wav_file_list)} files sent.")


def main(args: argparse.Namespace):
    """
    Main entrypoint: validates WAV files and starts the generation with the specified speech
    processor.
    """
    setup_metrics_logger(SimpleNamespace(**{
        "enabled": True,
        "filename": args.metrics_log_file
    }))
    LOGGER.info(f"Loading speech processor from {args.speech_processor_config}")
    speech_processor_config = yaml_config(args.speech_processor_config)
    LOGGER.info(f"Using as speech processor: {speech_processor_config.type}")
    speech_processor_loading_time = time.time()
    speech_processor = build_speech_processor(speech_processor_config)
    speech_processor_loading_time = time.time() - speech_processor_loading_time
    LOGGER.info(f"Loaded speech processor in {speech_processor_loading_time:.3f} seconds")
    METRICS_LOGGER.info(json.dumps({
        "model_loading_time": speech_processor_loading_time,
    }))
    wav_files = load_wav_file_list(args.wav_list_file)
    run_inference(speech_processor, wav_files, args.tgt_lang, args.src_lang)


def cli_main():
    """
    Simulstream evaluation command-line interface (CLI) entry point. This script processed the
    specified wav files with the configured speech processor and can be used to get the metrics
    log file to evaluate the quality and latency of the speech processor.

    This function parses command-line arguments and starts the asynchronous :func:`main` routine.

    Example usage::

        $ python inference.py --speech-processor-config config/speech.yaml \\
              --wav-list-file wav_files.txt --tgt-lang it --src-lang en

    Command-line arguments:

    - ``--server-config`` (str, optional): Path to the server configuration file
      (default: ``config/server.yaml``).
    - ``--speech-processor-config`` (str, required): Path to the speech processor configuration
      file.
    """
    LOGGER.info(f"Simulstream version: {simulstream.__version__}")
    parser = argparse.ArgumentParser("simulstream_inference")
    parser.add_argument("--speech-processor-config", type=str, required=True)
    parser.add_argument(
        "--wav-list-file",
        required=True,
        help="Path to text file containing list of WAV file paths")
    parser.add_argument(
        "--tgt-lang",
        default=None,
        help="Target language (if needed, its effect depends on the speech processor used by the "
             "server).")
    parser.add_argument(
        "--src-lang",
        default=None,
        help="Source language (if needed, its effect depends on the speech processor used by the "
             "server).")
    parser.add_argument(
        "--metrics-log-file",
        default="metrics.json",
        help="Path where to write the metrics log file.")
    main(parser.parse_args())


if __name__ == "__main__":
    cli_main()
