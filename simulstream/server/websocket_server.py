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

import websockets
from websockets.asyncio.server import serve, ServerConnection
import json

import simulstream
from simulstream.config import yaml_config
from simulstream.metrics.logger import setup_metrics_logger, METRICS_LOGGER
from simulstream.server.message_processor import MessageProcessor
from simulstream.server.speech_processors import build_speech_processor


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGER = logging.getLogger('simulstream.websocket_server')


def connection_handler_factory(
        speech_processor_config: SimpleNamespace) -> Callable[[ServerConnection], Awaitable[None]]:
    """
    Factory function that creates a connection handler for the WebSocket server.

    The returned connection handler routine will process audio and metadata messages sent by a
    single client over WebSocket.

    The handler:

    - Receives and buffering raw audio chunks (``bytes``).
    - Resamples audio to the system's :data:`~simulstream.server.speech_processors.SAMPLE_RATE`.
    - Processes audio incrementally with the configured
      :class:`~simulstream.server.speech_processors.SpeechProcessor`.
    - Handles textual messages for client metadata, language settings, and end-of-stream signals.
    - Sends back incremental processing results to the client in JSON format.

    :param speech_processor_config: Speech processor configuration form the specified YAML file.
    :type speech_processor_config: types.SimpleNamespace
    :return: An asynchronous WebSocket connection handler coroutine.
    :rtype: Callable[[websockets.asyncio.server.ServerConnection], Awaitable[None]]
    """

    async def handle_connection(websocket: ServerConnection) -> None:
        """
        Handles a single client WebSocket connection.

        This is the coroutine that processes incoming messages from a client:

        - If the message is binary (``bytes``), it is interpreted as raw audio data and
          buffered until a full chunk is ready for processing.
        - If the message is text (``str``), it is parsed as JSON metadata and can:

          - Set the input sample rate.
          - Set source and target languages for translation.
          - Log custom metadata to the metrics logger.
          - Indicate the end of the audio stream.

        At the end of the stream, any remaining audio is processed, the processor state is cleared,
        and an ``end_of_processing`` message is sent to the client.

        :param websocket: The WebSocket connection for the client.
        :type websocket: websockets.asyncio.server.ServerConnection
        """
        client_id = id(websocket)
        LOGGER.info(f"Client {client_id} connected")
        message_processor = MessageProcessor(
            client_id, build_speech_processor(speech_processor_config))

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    # in this case we are processing an audio chunk
                    incremental_output = message_processor.process_speech(message)
                    if incremental_output is not None:
                        await websocket.send(incremental_output.strings_to_json())
                elif isinstance(message, str):
                    # textual message are used to handle metadata
                    try:
                        data = json.loads(message)
                        if 'end_of_stream' in data:
                            incremental_output = message_processor.end_of_stream()
                            if incremental_output is not None:
                                await websocket.send(incremental_output.strings_to_json())
                            await websocket.send(json.dumps({'end_of_processing': True}))
                        else:
                            message_processor.process_metadata(data)
                    except Exception as e:
                        LOGGER.error(
                            f"Invalid string message: {message}. Error: {e}. Ignoring it.")
        except websockets.exceptions.ConnectionClosed:
            LOGGER.info(f"Client {client_id} disconnected.")
        except Exception as e:
            LOGGER.exception(f"Error: {e}")

    return handle_connection


async def main(args: argparse.Namespace):
    """
    Main entry point for running the WebSocket speech server.

    This function loads the server and speech processor configurations from YAML,
    initializes logging (including metrics logging), and starts the WebSocket server
    on the configured host and port.

    :param args: Parsed command-line arguments containing configuration file paths.
    :type args: argparse.Namespace
    """
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
            connection_handler_factory(speech_processor_config),
            server_config.hostname,
            server_config.port,
            ping_timeout=None) as server:
        await server.serve_forever()


def cli_main():
    """
    Simulstream WebSocket server command-line interface (CLI) entry point.

    This function parses command-line arguments and starts the asynchronous :func:`main` routine.

    Example usage::

        $ python websocket_server.py --server-config config/server.yaml \\
              --speech-processor-config config/speech.yaml

    Command-line arguments:

    - ``--server-config`` (str, optional): Path to the server configuration file
      (default: ``config/server.yaml``).
    - ``--speech-processor-config`` (str, required): Path to the speech processor configuration
      file.
    """
    LOGGER.info(f"Websocket server version: {simulstream.__version__}")
    parser = argparse.ArgumentParser("websocket_simul_server")
    parser.add_argument("--server-config", type=str, default="config/server.yaml")
    parser.add_argument("--speech-processor-config", type=str, required=True)
    args = parser.parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli_main()
