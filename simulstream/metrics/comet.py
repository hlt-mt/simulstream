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
import sys
from typing import Dict, List

import mweralign

import simulstream
from simulstream.config import yaml_config
from simulstream.metrics.readers import LogReader, ReferencesReader, YamlReferenceReader


logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGER = logging.getLogger('simulstream.comet')


def score_st(
        hypo_dict: Dict[str, str],
        ref_dict: Dict[str, List[str]],
        transcr_dict: Dict[str, List[str]]) -> float:
    """
    Computes COMET.
    """
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        sys.exit("Please install comet first with `pip install unbabel-comet==2.2.4`.")

    comet_data = []
    for name, ref_lines in ref_dict.items():
        src_lines = transcr_dict[name]
        assert len(ref_lines) == len(src_lines), \
            f"Reference ({name}) has mismatched number of target ({len(ref_lines)}) " \
            f"and source lines ({len(src_lines)})"
        hypo = hypo_dict[name]

        resegm_hypos = mweralign.align_texts("\n".join(ref_lines), hypo).split("\n")

        assert len(ref_lines) == len(resegm_hypos), \
            f"Reference ({name}) has mismatched number of target ({len(ref_lines)}) " \
            f"and resegmented lines ({len(resegm_hypos)})"
        for hyp, ref, src in zip(resegm_hypos, ref_lines, src_lines):
            comet_data.append({
                "src": src.strip(),
                "mt": hyp.strip(),
                "ref": ref.strip()
            })
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    model.eval()
    model_output = model.predict(comet_data, batch_size=8, gpus=1)
    return model_output.system_score


def main(args: argparse.Namespace):
    LOGGER.info(f"Loading evaluation configuration from {args.eval_config}")
    eval_config = yaml_config(args.eval_config)
    log_reader = LogReader(eval_config, args.log_file)
    if args.audio_definition is not None:
        assert len(args.references) == 1, \
            "When audio definition is provided, only one reference file should be provided."
        assert len(args.transcripts) == 1, \
            "When audio definition is provided, only one transcript file should be provided."
        reference_reader = YamlReferenceReader(args.audio_definition, args.references[0])
        transcripts_reader = YamlReferenceReader(args.audio_definition, args.transcripts[0])
    else:
        reference_reader = ReferencesReader(args.references)
        transcripts_reader = ReferencesReader(args.transcripts)
    comet_score = score_st(
        log_reader.final_outputs(),
        reference_reader.get_reference_texts(),
        transcripts_reader.get_reference_texts())
    print(f"COMET score: {comet_score}")


def cli_main():
    LOGGER.info(f"Simulstream version: {simulstream.__version__}")
    parser = argparse.ArgumentParser("comet")
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
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
