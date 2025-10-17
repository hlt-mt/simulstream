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
import asyncio
import json
import logging
import wave
from typing import Tuple, Optional, List

import numpy as np
import websockets
import os
import contextlib


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGER = logging.getLogger('simulstream.wav_reader_client')


def float32_to_int16(audio_data: np.ndarray) -> np.ndarray:
    """Convert a NumPy array of float32 audio samples to int16 PCM format."""
    audio_data = np.clip(audio_data * 2 ** 15, -32768, 32767)
    return audio_data.astype(np.int16)


def read_wav_file(filename: str) -> Tuple[int, np.ndarray]:
    """
    Read a WAV file and return its sample rate and audio data.

    Args:
        filename (str): Path to the WAV file.

    Returns:
        tuple[int, np.ndarray]: Sample rate and mono audio data as int16 array.

    Raises:
        ValueError: If the sample width is unsupported.
        AssertionError: If the file contains more than one channel.
    """
    with contextlib.closing(wave.open(filename, 'rb')) as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        raw_data = wf.readframes(num_frames)

        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.float32
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        data = np.frombuffer(raw_data, dtype=dtype)

        if sample_width == 4:
            data = float32_to_int16(data)

        assert num_channels == 1, "Currently ony 1 channel is supported"

        return sample_rate, data


async def send_audio(
        websocket: websockets.ClientConnection,
        sample_rate: int,
        data: np.ndarray,
        chunk_duration_ms: int = 100):
    """
    Stream audio data in fixed-size chunks over a WebSocket connection.

    Args:
        websocket (websockets.ClientConnection): Active WebSocket connection.
        sample_rate (int): Audio sample rate (Hz).
        data (np.ndarray): Audio samples as int16 array.
        chunk_duration_ms (int): Duration of each chunk in milliseconds.
    """
    samples_per_chunk = int(sample_rate * chunk_duration_ms / 1000.0)
    i = 0
    for i in range(0, len(data), samples_per_chunk):
        await websocket.send(data[i:i + samples_per_chunk].tobytes())
    # send last part of the audio
    if i < len(data):
        await websocket.send(data[i:].tobytes())


async def stream_wav_files(
        uri: str,
        wav_file_list: List[str],
        chunk_duration_ms: int = 100,
        tgt_lang: Optional[str] = None,
        src_lang: Optional[str] = None):
    """
    Stream multiple WAV files sequentially to a WebSocket server.

    For each file:
      - Sends metadata (sample rate, filename, optional languages).
      - Streams audio in chunks.
      - Sends an end-of-stream marker.
      - Waits for server confirmation before proceeding.

    Args:
        uri (str): WebSocket server URI.
        wav_file_list (list[str]): Paths to WAV files.
        chunk_duration_ms (int): Chunk size in milliseconds.
        tgt_lang (str | None): Target language code (e.g., "en").
        src_lang (str | None): Source language code (e.g., "en").
    """
    for wav_file in wav_file_list:
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
        async with websockets.connect(uri, ping_timeout=None) as websocket:
            await websocket.send(json.dumps(metadata))
            await send_audio(websocket, sample_rate, data, chunk_duration_ms)
            await websocket.send(json.dumps({"end_of_stream": True}))
            while True:
                response = await websocket.recv()
                LOGGER.debug(response)
                if 'end_of_processing' in response:
                    break
    LOGGER.info(f"All {len(wav_file_list)} files sent.")


def load_wav_file_list(list_file_path: str) -> List[str]:
    """
    Load a list of WAV file paths from a text file.

    Args:
        list_file_path (str): Path to a text file, one WAV file path per line.

    Returns:
        list[str]: Absolute file paths of WAV files.
    """
    basedir = os.path.dirname(list_file_path)
    with open(list_file_path, 'r') as f:
        wav_files = [basedir + '/' + line.strip() for line in f if line.strip()]
    if not wav_files:
        LOGGER.error("No valid WAV files found in the list.")
        exit(1)
    else:
        assert all(os.path.isfile(f) for f in wav_files), "Invalid wav file in the list."
    return wav_files


async def main(args: argparse.Namespace):
    """Main entrypoint: validates WAV files and starts streaming."""
    wav_files = load_wav_file_list(args.wav_list_file)
    await stream_wav_files(
        args.uri, wav_files, args.chunk_duration_ms, args.tgt_lang, args.src_lang)


def cli_main():
    """
    Simulstream WebSocket client command-line interface (CLI) entry point.

    This script implements a simple WebSocket client that streams audio data from a list of WAV
    files to a server for processing (e.g., speech recognition or translation). It reads WAV files,
    converts them into fixed-size chunks, and sends them asynchronously over a WebSocket
    connection.

    Example usage::

        $ python wav_reader_client.py --uri ws://localhost:8000/ --wav-list-file wav_files.txt \\
              --tgt-lang it --src-lang en

    Command-line arguments:

    - ``--uri``: WebSocket server URI (e.g., ``ws://localhost:8000/``).
    - ``--wav-list-file``: Path to a text file containing one WAV file path per line.
    - ``--chunk-duration-ms``: Duration of each audio chunk sent to the server (ms). Default: 100.
    - ``--tgt-lang``: Optional target language.
    - ``--src-lang``: Optional source language.
    """
    parser = argparse.ArgumentParser(description="Websocket client for WAV files.")
    parser.add_argument(
        "--uri",
        required=True,
        help="WebSocket server URI (e.g., ws://localhost:8000/)")
    parser.add_argument(
        "--wav-list-file",
        required=True,
        help="Path to text file containing list of WAV file paths")
    parser.add_argument(
        "--chunk-duration-ms",
        default=100,
        type=int,
        help="Size of the each chunk sent to the server in milliseconds (default: 100)")
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
    asyncio.run(main(parser.parse_args()))


if __name__ == "__main__":
    cli_main()
