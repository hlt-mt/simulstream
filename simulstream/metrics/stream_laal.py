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
from typing import Dict, List, Tuple

import mweralign

import simulstream
from simulstream.config import yaml_config
from simulstream.metrics.readers import LogReader, YamlReferenceReader, \
    ReferenceSentenceDefinition, OutputWithDelays, text_items


_VERSION = "2.0"
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
)
LOGGER = logging.getLogger('simulstream.streamlaal')


def laal(delays: List[float], source_length: float, target_length: int) -> float:
    """
    Function to compute Length Adaptive Average Lagging (LAAL) on one sentence as proposed in
    `CUNI-KIT System for Simultaneous Speech Translation Task at IWSLT 2022
    <https://arxiv.org/abs/2204.06028>`_ and
    `Length-Adaptive Average Lagging for Simultaneous Speech Translation
    <https://arxiv.org/abs/2206.05807>`_.
    It is the original Average Lagging as proposed in
    `Controllable Latency using Prefix-to-Prefix Framework
    <https://arxiv.org/abs/1810.08398>`_
    but is robust to the length difference between the hypothesis and reference.

    The implementation is derived by that available in SimulEval (see `latency_scorer.py` in
    `https://github.com/facebookresearch/SimulEval/).

    Returns:
        float: the latency score on one sentence.
    """
    if delays[0] > source_length:
        return delays[0]

    LAAL = 0
    gamma = max(len(delays), target_length) / source_length
    tau = 0
    for t_minus_1, d in enumerate(delays):
        LAAL += d - t_minus_1 / gamma
        tau = t_minus_1 + 1

        if d >= source_length:
            break
    LAAL /= tau
    return LAAL


def split_delays_by_segmented_text(
        delays: List[float], segmented_text: List[str], latency_unit: str):
    segmented_delays = []
    index = 0

    for segment in segmented_text:
        segment_len = len(text_items(segment, latency_unit))
        segmented_delays.append(delays[index:index + segment_len])
        index += segment_len
    assert len(delays) == index, f"Index {index} should have reached end of delays ({len(delays)})"
    return segmented_delays


def score_streamlaal(
        hypo_dict: Dict[str, OutputWithDelays],
        ref_dict: Dict[str, List[ReferenceSentenceDefinition]],
        latency_unit: str) -> Tuple[float, float]:
    """
    Computes StreamLAAL version 2.0, as proposed in
    `StreamAtt: Direct Streaming Speech-to-Text Translation with Attention-based
    Audio History Selection <https://aclanthology.org/2024.acl-long.202.pdf>`_.


    Then main difference with version 1 is the different segmentation of
    the text (uses levenshtein aligner from SubER instead of mwerSegmenter).-
    """

    streamlaal_ideal_scores = []
    streamlaal_computational_aware_scores = []
    skipped_sentences = 0
    for name, ref_sentences_def in ref_dict.items():
        hypo_with_delays = hypo_dict[name]

        resegmented_hypos = mweralign.align_texts(
            "\n".join([sentence_def.content for sentence_def in ref_sentences_def]),
            hypo_with_delays.final_text).split("\n")

        assert len(resegmented_hypos) == len(ref_sentences_def), \
            f"Reference ({name}) has mismatched number of target ({len(ref_sentences_def)}) " \
            f"and resegmented lines ({len(resegmented_hypos)})"

        resegmented_ideal_delays = split_delays_by_segmented_text(
            hypo_with_delays.ideal_delays, resegmented_hypos, latency_unit)
        resegmented_computational_delays = split_delays_by_segmented_text(
            hypo_with_delays.computational_aware_delays, resegmented_hypos, latency_unit)

        for i in range(len(ref_sentences_def)):
            delays_from_sentence_start = [
                delay - ref_sentences_def[i].start_time
                for delay in resegmented_ideal_delays[i]]
            ca_delays_from_sentence_start = [
                delay - ref_sentences_def[i].start_time
                for delay in resegmented_computational_delays[i]]
            assert len(delays_from_sentence_start) == len(ca_delays_from_sentence_start)
            target_length = len(text_items(ref_sentences_def[i].content, latency_unit))
            if len(delays_from_sentence_start) > 0:
                streamlaal_ideal_scores.append(laal(
                    delays_from_sentence_start,
                    ref_sentences_def[i].duration,
                    target_length))
                streamlaal_computational_aware_scores.append(laal(
                    ca_delays_from_sentence_start,
                    ref_sentences_def[i].duration,
                    target_length))
            else:
                skipped_sentences += 1

    if skipped_sentences > 0:
        LOGGER.warning(
            f"{skipped_sentences} sentences have been skipped in latency computation as they were "
            "empty")
    stream_laal_ideal = sum(streamlaal_ideal_scores) / len(streamlaal_ideal_scores)
    stream_laal_computational = \
        sum(streamlaal_computational_aware_scores) / len(streamlaal_computational_aware_scores)
    return stream_laal_ideal, stream_laal_computational


def main(args: argparse.Namespace):
    LOGGER.info(f"Loading evaluation configuration from {args.eval_config}")
    eval_config = yaml_config(args.eval_config)
    log_reader = LogReader(eval_config, args.log_file, latency_unit=args.latency_unit)
    reference_reader = YamlReferenceReader(args.audio_definition, args.reference)
    streamlaal_ideal, streamlaal_computational_aware = score_streamlaal(
        log_reader.final_outputs_and_latencies(),
        reference_reader.references,
        args.latency_unit)
    scores = {
        "ideal": streamlaal_ideal,
        "computational_aware": streamlaal_computational_aware
    }
    print(f"LAAL {_VERSION} scores (in seconds): {scores}")


def cli_main():
    LOGGER.info(f"Simulstream version: {simulstream.__version__}")
    parser = argparse.ArgumentParser("stream_laal")
    parser.add_argument(
        "--eval-config", type=str, required=True,
        help="Path to the yaml config file containing information about the tokenizer to be used.")
    parser.add_argument(
        "--log-file", type=str, required=True,
        help="Path to the log file with the metrics to be used for the evaluation.")
    parser.add_argument(
        "--reference", "-r", required=True, type=str,
        help="Path to the textual file containing segment-level references stored line by line.")
    parser.add_argument(
        "--audio-definition", "-a", type=str, required=True,
        help="Path to the yaml file containing the segment-level audio information.")
    parser.add_argument(
        "--latency-unit", choices=["word", "char"], default="word",
        help="Whether to computed latency based on words or characters. Default: word.")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
