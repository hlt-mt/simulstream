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
import logging

import simulstream
from simulstream.config import yaml_config
from simulstream.metrics.readers import LogReader, YamlReferenceReader
from simulstream.metrics.scorers.latency import LATENCY_SCORER_REGISTRY, LatencyScorer, LatencyScoringSample


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGER = logging.getLogger('simulstream.score_latency')


def main(scorer_cls: type[LatencyScorer], args: argparse.Namespace):
    LOGGER.info(f"Loading evaluation configuration from {args.eval_config}")
    eval_config = yaml_config(args.eval_config)
    LOGGER.info(f"Reading log file ({args.log_file})")
    log_reader = LogReader(eval_config, args.log_file, latency_unit=args.latency_unit)

    LOGGER.info(f"Building latency scorer {args.scorer}")
    scorer = scorer_cls(args)

    LOGGER.info(
        f"Reading audio definition ({args.audio_definition}), and reference ({args.reference})")
    references = None
    if scorer.requires_reference():
        reference_reader = YamlReferenceReader(args.audio_definition, args.reference)
        references = reference_reader.references

    output_with_latency = log_reader.final_outputs_and_latencies()

    if references is not None:
        audio_files = references.keys()
    else:
        audio_files = output_with_latency.keys()

    samples = []
    for audio_file in audio_files:
        reference = references[audio_file] if references is not None else None
        samples.append(LatencyScoringSample(
            audio_file, output_with_latency[audio_file], reference))

    score = scorer.score(samples)
    print(f"Latency scores (in seconds): {score}")


def cli_main():
    LOGGER.info(f"Simulstream version: {simulstream.__version__}")
    parser = argparse.ArgumentParser("score_latency")
    parser.add_argument(
        "--eval-config", type=str, required=True,
        help="Path to the yaml config file containing information about the tokenizer to be used.")
    parser.add_argument(
        "--log-file", type=str, required=True,
        help="Path to the log file with the metrics to be used for the evaluation.")
    parser.add_argument(
        "--reference", "-r", type=str,
        help="Path to the textual file containing segment-level references stored line by line.")
    parser.add_argument(
        "--audio-definition", "-a", type=str, required=True,
        help="Path to the yaml file containing the segment-level audio information.")
    parser.add_argument(
        "--latency-unit", choices=["word", "char"], default="word",
        help="Whether to computed latency based on words or characters. Default: word.")
    parser.add_argument("--scorer", choices=LATENCY_SCORER_REGISTRY.keys(), required=True)
    args, _ = parser.parse_known_args()

    # build full parser with scorer-specific args
    parser = argparse.ArgumentParser(parents=[parser], add_help=False)
    scorer_cls = LATENCY_SCORER_REGISTRY[args.scorer]
    scorer_cls.add_arguments(parser)

    # parse new arguments
    args = parser.parse_args()

    main(scorer_cls, args)


if __name__ == "__main__":
    cli_main()
