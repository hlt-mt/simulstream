FROM python:3.11

WORKDIR /home/simulstream_agent

COPY --from=simulstream_base config ./config
COPY --from=simulstream_base simulstream ./simulstream
COPY --from=simulstream_base pyproject.toml setup.py ./

# First install torch with GPU support, otherwise the CPU version gets installed
# update this to your CUDA version (this is for CUDA 13.0)
RUN pip install torch --index-url https://download.pytorch.org/whl/cu130
# Install the dependencies for your model and install simulstream
RUN pip install -e .[canary]


EXPOSE 8080

CMD [ \
    "python", \
    "-m", \
    "simulstream.server.speech_processors.remote.http_speech_processor_server", \
    "--speech-processor-config", \
    "config/canary_sliding_window_retranslation.yaml" \
 ]
