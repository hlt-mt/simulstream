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

import simulstream
from simulstream.config import yaml_config
from simulstream.metrics.reader import LogReader, ReferenceReader
from simulstream.metrics.resegmenter import levenshtein_align_hypothesis_to_reference

try:
    from comet import download_model, load_from_checkpoint
except ImportError:
    sys.exit("Please install comet first with `pip install unbabel-comet==2.2.4`.")

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
    comet_data = []
    for name, ref_lines in ref_dict.items():
        src_lines = transcr_dict[name]
        assert len(ref_lines) == len(src_lines), \
            f"Reference ({name}) has mismatched number of target ({len(ref_lines)}) " \
            f"and source lines ({len(src_lines)})"
        hypo = hypo_dict[name]

        resegm_hypos = levenshtein_align_hypothesis_to_reference([hypo], ref_lines)

        assert len(ref_lines) == len(resegm_hypos), \
            f"Reference ({name}) has mismatched number of target ({len(resegm_hypos)}) " \
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
    reference_reader = ReferenceReader(args.references)
    transcripts_reader = ReferenceReader(args.transcripts)
    comet_score = score_st(
        log_reader.final_outputs(), reference_reader.get_all(), transcripts_reader.get_all())
    print(f"COMET score: {comet_score}")


def cli_main():
    LOGGER.info(f"Simulstream version: {simulstream.__version__}")
    parser = argparse.ArgumentParser("comet")
    parser.add_argument("--eval-config", type=str, required=True)
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument("--references", nargs="+", type=str, required=True)
    parser.add_argument("--transcripts", nargs="+", type=str, required=True)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
