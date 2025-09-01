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
from simulstream.metrics.readers import LogReader, ReferencesReader, YamlReferenceReader
from simulstream.metrics.scorers.quality import QUALITY_SCORER_REGISTRY, QualityScorer, QualityScoringSample


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGER = logging.getLogger('simulstream.score_quality')


def main(scorer_cls: type[QualityScorer], args: argparse.Namespace):
    LOGGER.info(f"Loading evaluation configuration from {args.eval_config}")
    eval_config = yaml_config(args.eval_config)
    log_reader = LogReader(eval_config, args.log_file)

    LOGGER.info(f"Building scorer class for {args.scorer}")
    scorer = scorer_cls(args)

    LOGGER.info("Reading source and reference definition")
    reference_reader = None
    transcripts_reader = None
    if args.audio_definition is not None:
        if scorer.requires_reference():
            assert len(args.references) == 1, \
                "When audio definition is provided, only one reference file should be provided."
            reference_reader = YamlReferenceReader(args.audio_definition, args.references[0])
        if scorer.requires_source():
            assert len(args.transcripts) == 1, \
                "When audio definition is provided, only one transcript file should be provided."
            transcripts_reader = YamlReferenceReader(args.audio_definition, args.transcripts[0])
    else:
        if scorer.requires_reference():
            reference_reader = ReferencesReader(args.references)
        if scorer.requires_source():
            transcripts_reader = ReferencesReader(args.transcripts)

    hypothesis_dictionary = log_reader.final_outputs()
    transcript_dictionary = None
    reference_dictionary = None
    audio_files_to_score = None
    if transcripts_reader is not None:
        transcript_dictionary = transcripts_reader.get_reference_texts()
        audio_files_to_score = transcript_dictionary.keys()
    if reference_reader is not None:
        reference_dictionary = reference_reader.get_reference_texts()
        audio_files_to_score = reference_dictionary.keys()

    scoring_samples = []
    for audio_name in audio_files_to_score:
        transcript = None
        if transcript_dictionary is not None:
            transcript = transcript_dictionary[audio_name]
        reference = None
        if reference_dictionary is not None:
            reference = reference_dictionary[audio_name]
        if transcript is not None and reference is not None:
            assert len(reference) == len(transcript), \
                f"Reference ({audio_name}) has mismatched number of target ({len(reference)}) " \
                f"and source lines ({len(transcript)})"

        scoring_samples.append(
            QualityScoringSample(audio_name, hypothesis_dictionary[audio_name], reference, transcript))

    LOGGER.info("Scoring outputs")
    score = scorer.score(scoring_samples)

    print(f"{args.scorer} score: {score}")


def cli_main():
    LOGGER.info(f"Simulstream version: {simulstream.__version__}")
    parser = argparse.ArgumentParser("score_quality")
    parser.add_argument(
        "--eval-config", type=str, required=True,
        help="Path to the yaml config file containing information about the tokenizer to be used.")
    parser.add_argument(
        "--log-file", type=str, required=True,
        help="Path to the log file with the metrics to be used for the evaluation.")
    parser.add_argument(
        "--references", nargs="+", type=str, required=True,
        help="Path to the textual files containing references. If `--audio-definition` is "
             "specified, this should be a single file containing all the lines of the audios in "
             "the reference, which should be of the same length of the audio definition. "
             "Otherwise, this should be a list of files, where each contains the lines "
             "corresponding to an audio file.")
    parser.add_argument(
        "--transcripts", nargs="+", type=str, required=True,
        help="Path to the textual files containing transcripts. If `--audio-definition` is "
             "specified, this should be a single file containing all the lines of the audios in "
             "the reference, which should be of the same length of the audio definition. "
             "Otherwise, this should be a list of files, where each contains the lines "
             "corresponding to an audio file.")
    parser.add_argument(
        "--audio-definition", "-a", type=str, default=None,
        help="Path to the yaml file containing the segment-level audio information.")
    parser.add_argument("--scorer", choices=QUALITY_SCORER_REGISTRY.keys(), required=True)
    args, _ = parser.parse_known_args()

    # build full parser with scorer-specific args
    parser = argparse.ArgumentParser(parents=[parser], add_help=False)
    scorer_cls = QUALITY_SCORER_REGISTRY[args.scorer]
    scorer_cls.add_arguments(parser)

    # parse new arguments
    args = parser.parse_args()

    main(scorer_cls, args)


if __name__ == "__main__":
    cli_main()
