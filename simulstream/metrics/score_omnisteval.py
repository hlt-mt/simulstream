import argparse
import logging
import os
from typing import Optional

import omnisteval
import sacrebleu

import simulstream
from omnisteval.io import (
    load_resegmentation_inputs as load_inputs,
    dump_instances_jsonl,
    dump_scores_tsv,
    format_report,
)
from omnisteval import resegment
from omnisteval import evaluate_instances

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,
)
LOGGER = logging.getLogger("simulstream.score_omnisteval")


def _build_settings(
    source_sentences_file: str,
    audio_definition: str,
    reference: str,
    log_file: str,
    latency_unit: str,
    eval_config: str,
    moses_tokenizer_lang: Optional[str],
    bleu_tokenizer: str,
    comet: bool,
    comet_model: str,
):
    """
    Build a dictionary of settings to log alongside the scores.
    This is not used for the actual scoring, but can be helpful for record-keeping and debugging.
    Keys match OmniSTEval settings where applicable.
    """

    return {
        "Hypothesis": log_file,
        "Hypothesis format": "simulstream",
        "Reference": reference,
        "Source sentences": source_sentences_file or "none",
        "Eval config": eval_config,
        "Segmentation": audio_definition,
        "Seg. type": "speech",
        "Language": moses_tokenizer_lang or "none",
        "BLEU tokenizer": bleu_tokenizer,
        "Char-level": "yes" if latency_unit == "char" else "no",
        "Offset delays": "no",
        "Fix CA emissions": "no",
        "COMET model": comet_model if comet else "none",
        "OmniSTEval version": omnisteval.__version__,
        "Simulstream version": simulstream.__version__,
    }


def main(
    source_sentences_file: str,
    audio_definition: str,
    reference: str,
    log_file: str,
    latency_unit: str,
    eval_config: str,
    output_folder: str,
    moses_tokenizer_lang: Optional[str],
    bleu_tokenizer: str,
    comet: bool,
    comet_model: str,
):
    LOGGER.info(f"Simulstream version: {simulstream.__version__}")

    LOGGER.info("Loading evaluation config and log file...")

    source_sentences = None
    if source_sentences_file is not None:
        with open(source_sentences_file, "r", encoding="utf-8") as f:
            source_sentences = [line.strip() for line in f]

    LOGGER.info("Loading resegmentation inputs for OmniSTEval...")
    ref_words, hyp_words, segmentation, ref_sentences, all_have_emission_ca = load_inputs(
        audio_definition,
        None,
        reference,
        log_file,
        hypothesis_format="simulstream",
        char_level=(latency_unit == "char"),
        offset_delays=False,
        fix_emission_ca_flag=False,
        simulstream_config_file=eval_config,
    )

    # suppress mosestokenizer INFO logs which are very verbose
    logging.getLogger("mosestokenizer").setLevel(logging.WARNING)

    LOGGER.info("Running resegmentation with OmniSTEval...")
    instances, instances_dict = resegment(
        ref_words=ref_words,
        hyp_words=hyp_words,
        segmentation=segmentation,
        ref_sentences=ref_sentences,
        char_level=(latency_unit == "char"),
        lang=moses_tokenizer_lang,
        has_emission_timestamps=all_have_emission_ca,
    )

    LOGGER.info("Computing metrics...")
    scores = evaluate_instances(
        instances=instances,
        compute_quality=True,
        compute_latency=True,
        is_longform=True,
        bleu_tokenizer=bleu_tokenizer,
        all_have_emission_ca=all_have_emission_ca,
        fix_emission_ca_flag=False,
        compute_comet=comet,
        comet_model=comet_model,
        source_sentences=source_sentences,
    )

    settings = _build_settings(
        source_sentences_file=source_sentences_file,
        audio_definition=audio_definition,
        reference=reference,
        log_file=log_file,
        latency_unit=latency_unit,
        eval_config=eval_config,
        moses_tokenizer_lang=moses_tokenizer_lang,
        bleu_tokenizer=bleu_tokenizer,
        comet=comet,
        comet_model=comet_model,
    )
    report = format_report("Longform evaluation (with resegmentation)", settings, scores)
    LOGGER.info(f"\n{report}")

    if output_folder is not None:
        dump_instances_jsonl(instances_dict, output_folder)
        dump_scores_tsv(scores, output_folder, is_longform=True)
        with open(
            os.path.join(output_folder, "evaluation_report.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(report)


def cli_main():
    parser = argparse.ArgumentParser(
        "run_omnisteval",
        description="Score streaming translation outputs using OmniSTEval.",
    )
    parser.add_argument("--eval-config", type=str, required=True)
    parser.add_argument("--log-file", type=str, required=True)
    parser.add_argument(
        "--audio-definition",
        "-a",
        type=str,
        required=True,
        help="Path to the yaml file containing the segment-level audio information.",
    )
    parser.add_argument(
        "--reference",
        "-r",
        type=str,
        required=True,
        help="Path to the textual file containing segment-level references stored line by line.",
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        default=None,
        help="Optional output folder for OmniSTEval artifacts.",
    )
    parser.add_argument(
        "--latency-unit",
        choices=["word", "char"],
        default="word",
        help="Whether to compute stats based on words or characters. Default: word.",
    )
    parser.add_argument(
        "--bleu-tokenizer",
        choices=sacrebleu.metrics.METRICS["BLEU"].TOKENIZERS,
        default=sacrebleu.metrics.METRICS["BLEU"].TOKENIZER_DEFAULT,
    )
    parser.add_argument(
        "--moses-tokenizer-lang",
        type=str,
        default="en",
        help='Language code for Moses tokenizer if BLEU tokenizer is set to "13a". Default: en.',
    )
    parser.add_argument(
        "--comet-model",
        type=str,
        default="Unbabel/wmt22-comet-da",
        help=(
            "Name or path of the COMET model to use for quality estimation when --comet is "
            "enabled. Default: Unbabel/wmt22-comet-da."
        ),
    )
    parser.add_argument("--comet", action="store_true", help="Enable COMET scoring.")
    parser.add_argument("--source-sentences-file", type=str, default=None)
    args = parser.parse_args()

    main(
        audio_definition=args.audio_definition,
        reference=args.reference,
        log_file=args.log_file,
        eval_config=args.eval_config,
        output_folder=args.output_folder,
        latency_unit=args.latency_unit,
        bleu_tokenizer=args.bleu_tokenizer,
        moses_tokenizer_lang=args.moses_tokenizer_lang,
        comet=args.comet,
        comet_model=args.comet_model,
        source_sentences_file=args.source_sentences_file,
    )


if __name__ == "__main__":
    cli_main()
