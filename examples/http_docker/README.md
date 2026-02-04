# Example of Docker Speech Processor

This folder contains a [Dockerfile](./Dockerfile) that is a working example of how to build a Docker
containing a speech processor (e.g., required for IWSLT submissions).

The Docker can be built by running the following command **from the directory
containing this REAMDE file and the Dockerfile**:

```shell
docker build --build-context simulstream_base=../.. -t http_speech_processor .
```

You can replace `http_speech_processor` with the name you want to give to your
docker image. Then, you can run the docker image with:

```shell
docker run --rm --gpus=all -p 8080:8080 http_speech_processor
```

And then, you can use `simulstream` setting the proxy HTTP processor to access your
dockerized speech processor, e.g., by running:

```shell
simulstream_inference --speech-processor-config config/http_proxy_processor.yaml \
   --wav-list-file $YOUR_TXT_FILE \
   --tgt-lang $TGT_LANG --src-lang $SRC_LANG \
   --metrics-log-file $YOUR_OUTPUT_JSONL_FILE
```

Please notice that this [Dockerfile example](./Dockerfile) runs a Canary sliding window speech processor.
You may want to edit the Dockerfile to include your code, including your custom speech processor,
your speech processor configuration file, and your requirements.
