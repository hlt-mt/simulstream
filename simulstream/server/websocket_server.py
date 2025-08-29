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
import logging
import time
from types import SimpleNamespace
from typing import Callable, Awaitable

import librosa
import numpy as np
import websockets
from websockets.asyncio.server import serve, ServerConnection
import json

import simulstream
from simulstream.config import yaml_config
from simulstream.metrics.logger import setup_metrics_logger, METRICS_LOGGER
from simulstream.server.speech_processors import build_speech_processor
from simulstream.server.speech_processors import SpeechProcessor, SAMPLE_RATE


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGER = logging.getLogger('simulstream.websocket_server')


def connection_handler_factory(
        server_config: SimpleNamespace,
        speech_processor_config: SimpleNamespace) -> Callable[[ServerConnection], Awaitable[None]]:
    """
    Returns a connection handler function that has in scope the given arguments.
    """

    async def process_audio(
            speech_processor: SpeechProcessor,
            client_id: int,
            audio_data: bytes,
            sample_rate: int,
            processed_audio_seconds: float) -> str:
        start_time = time.time()
        int16_waveform = np.frombuffer(audio_data, dtype=np.int16)
        float32_waveform = int16_waveform.astype(np.float32) / 2**15
        if sample_rate != SAMPLE_RATE:
            float32_waveform = librosa.resample(
                float32_waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
        incremental_output = speech_processor.process_chunk(float32_waveform)
        end_time = time.time()
        METRICS_LOGGER.info(json.dumps({
            "id": client_id,
            "total_audio_processed": processed_audio_seconds,
            "computation_time": end_time - start_time,
            "generated_tokens": incremental_output.new_tokens,
            "deleted_tokens": incremental_output.deleted_tokens,
        }))
        return incremental_output.strings_to_json()

    async def handle_connection(websocket: ServerConnection) -> None:
        """
        This is the method that process the connection of a client. It iterates over the messages
        received from the client and orchestrates how they are processed and the messages sent to
        the client.
        """
        client_id = id(websocket)
        client_buffer = b''
        processed_audio_seconds = 0
        LOGGER.info(f"Client {client_id} connected")
        sample_rate = SAMPLE_RATE
        speech_processor = build_speech_processor(speech_processor_config)

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # in this case we are processing an audio chunk
                    client_buffer += message
                    # we have SAMPLE_RATE * 2 bytes (int16) samples every second
                    buffer_len_seconds = len(client_buffer) / 2 / sample_rate
                    if buffer_len_seconds >= server_config.speech_processing_frequency:
                        processed_audio_seconds += buffer_len_seconds
                        response = await process_audio(
                            speech_processor,
                            client_id,
                            client_buffer,
                            sample_rate,
                            processed_audio_seconds)
                        await websocket.send(response)
                        client_buffer = b''
                elif isinstance(message, str):
                    # textual message are used to handle metadata
                    try:
                        data = json.loads(message)
                        if 'sample_rate' in data:
                            sample_rate = int(data['sample_rate'])
                        if 'target_lang' in data:
                            speech_processor.set_target_language(data["target_lang"])
                            LOGGER.debug(
                                f"Client {client_id} target language set to: "
                                f"{data['target_lang']}")
                        if 'source_lang' in data:
                            speech_processor.set_source_language(data["source_lang"])
                            LOGGER.debug(
                                f"Client {client_id} source language set to: "
                                f"{data['source_lang']}")
                        if 'metrics_metadata' in data:
                            METRICS_LOGGER.info(json.dumps({
                                "id": client_id,
                                "metadata": data["metrics_metadata"]
                            }))
                            LOGGER.debug(
                                f"Logged client {client_id} metrics metadata: "
                                f"{data['metrics_metadata']}")
                        if 'end_of_stream' in data:
                            if client_buffer:
                                # process remaining audio after last chunk
                                processed_audio_seconds += len(client_buffer) / 2 / sample_rate
                                response = await process_audio(
                                    speech_processor,
                                    client_id,
                                    client_buffer,
                                    sample_rate,
                                    processed_audio_seconds)
                                await websocket.send(response)
                            client_buffer = b''
                            speech_processor.clear()
                            await websocket.send(json.dumps({'end_of_processing': True}))
                    except Exception as e:
                        LOGGER.error(
                            f"Invalid string message: {message}. Error: {e}. Ignoring it.")
        except websockets.exceptions.ConnectionClosed:
            LOGGER.info(f"Client {client_id} disconnected.")
        except Exception as e:
            LOGGER.exception(f"Error: {e}")

    return handle_connection


async def main(args: argparse.Namespace):
    LOGGER.info(f"Loading server configuration from {args.server_config}")
    server_config = yaml_config(args.server_config)
    LOGGER.info(
        f"Metric logging is{'' if server_config.metrics.enabled else ' NOT'} enabled at "
        f"{server_config.metrics.filename}")
    setup_metrics_logger(server_config.metrics)
    LOGGER.info(f"Loading speech processor from {args.speech_processor_config}")
    speech_processor_config = yaml_config(args.speech_processor_config)
    LOGGER.info(f"Using as speech processor: {speech_processor_config.type}")
    speech_processor_loading_time = time.time()
    build_speech_processor(speech_processor_config)
    speech_processor_loading_time = time.time() - speech_processor_loading_time
    LOGGER.info(f"Loaded speech processor in {speech_processor_loading_time:.3f} seconds")
    METRICS_LOGGER.info(json.dumps({
        "model_loading_time": speech_processor_loading_time,
    }))
    LOGGER.info(f"Serving websocket server at {server_config.hostname}:{server_config.port}")
    async with serve(
            connection_handler_factory(server_config, speech_processor_config),
            server_config.hostname,
            server_config.port,
            ping_timeout=None) as server:
        await server.serve_forever()


def cli_main():
    LOGGER.info(f"Websocket server version: {simulstream.__version__}")
    parser = argparse.ArgumentParser("websocket_simul_server")
    parser.add_argument("--server-config", type=str, default="config/server.yaml")
    parser.add_argument("--speech-processor-config", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli_main()
