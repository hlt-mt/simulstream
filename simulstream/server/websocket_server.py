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
from contextlib import asynccontextmanager
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


class SpeechProcessorPool:
    """
    A pool of speech processors initialized at startup and made available to clients.

    Args:
        speech_processor_config (SimpleNamespace): configuration for the speech processors to
           create.
        size (int): number of speech processors to have in the pool.
        acquire_timeout (int): timeout (in seconds) for waiting the availability of a speech
           processor.
    """
    def __init__(
            self,
            speech_processor_config: SimpleNamespace,
            size: int,
            acquire_timeout: int):
        self.size = size
        self.acquire_timeout = acquire_timeout
        self.available = asyncio.Queue(maxsize=size)
        for _ in range(size):
            self.available.put_nowait(build_speech_processor(speech_processor_config))

    @asynccontextmanager
    async def acquire(self):
        """
        Acquire one process from the pool and release it automatically.

        Returns:
            SpeechProcessor: a speech processor available for usage.

        Raises:
            TimeoutError: if no speech processor is available within the configured timeout.
        """
        speech_processor = await asyncio.wait_for(
            self.available.get(), timeout=self.acquire_timeout)
        try:
            yield speech_processor
        finally:
            # Return worker to pool
            self.available.put_nowait(speech_processor)


def connection_handler_factory(
        speech_processor_pool: SpeechProcessorPool
) -> Callable[[ServerConnection], Awaitable[None]]:
    """
    Factory function that creates a connection handler for the WebSocket server.

    The returned connection handler routine will process audio and metadata messages sent by a
    single client over WebSocket.

    The handler receives client data (raw audio chunks and textual metadata) and forwards it to a
    message processor using a speech processor retrieved form the pool of the available ones.
    If no speech processor is available for the configured waiting time, the client connection is
    closed.
    The handler also sends incremental processing results back to the client in JSON format.

    Args:
        speech_processor_pool (SpeechProcessorPool): Pool of speech processors to use to handle
            client connections.

    Returns:
        Callable[[websockets.asyncio.server.ServerConnection], Awaitable[None]]: An asynchronous
            WebSocket connection handler coroutine.
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

        Args:
            websocket (websockets.asyncio.server.ServerConnection): The WebSocket connection for
                the client.
        """
        loop = asyncio.get_running_loop()
        client_id = id(websocket)
        LOGGER.info(f"Client {client_id} connected")
        try:
            async with speech_processor_pool.acquire() as speech_processor:
                message_processor = MessageProcessor(client_id, speech_processor)

                try:
                    async for message in websocket:
                        if isinstance(message, bytes):
                            # in this case we are processing an audio chunk
                            incremental_output = await loop.run_in_executor(
                                None, message_processor.process_speech, message)
                            if incremental_output is not None:
                                await websocket.send(incremental_output.strings_to_json())
                        elif isinstance(message, str):
                            # textual message are used to handle metadata
                            try:
                                data = json.loads(message)
                                if 'end_of_stream' in data:
                                    incremental_output = await loop.run_in_executor(
                                        None, message_processor.end_of_stream)
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
                finally:
                    message_processor.clear()
        except TimeoutError:
            LOGGER.error(f"Timeout waiting for a new processor for client {client_id}")
            await websocket.close()

    return handle_connection


async def main(args: argparse.Namespace):
    """
    Main entry point for running the WebSocket speech server.

    This function loads the server and speech processor configurations from YAML,
    initializes logging (including metrics logging), and starts the WebSocket server
    on the configured host and port.

    Args:
        args (argparse.Namespace): parsed command-line arguments with configuration file paths.
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
    speech_processors_pool = SpeechProcessorPool(
        speech_processor_config, server_config.pool_size, server_config.acquire_timeout)
    speech_processor_loading_time = time.time() - speech_processor_loading_time
    LOGGER.info(f"Loaded speech processor in {speech_processor_loading_time:.3f} seconds")
    METRICS_LOGGER.info(json.dumps({
        "model_loading_time": speech_processor_loading_time,
    }))
    LOGGER.info(f"Serving websocket server at {server_config.hostname}:{server_config.port}")
    async with serve(
            connection_handler_factory(speech_processors_pool),
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
