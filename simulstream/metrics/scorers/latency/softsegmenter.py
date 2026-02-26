# Copyright 2026 FBK

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
import unicodedata
from abc import abstractmethod
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Callable, List, Optional, Tuple

from mosestokenizer import MosesTokenizer

from simulstream.metrics.readers import ReferenceSentenceDefinition, OutputWithDelays, text_items
from simulstream.metrics.scorers.latency import LatencyScoringSample, LatencyScores
from simulstream.metrics.scorers.latency import ResegmentedLatencyScoringSample
from simulstream.metrics.scorers.latency.segmenter_based_scorer import SegmenterBasedScorer


LOGGER = logging.getLogger("simulstream.metrics.scorers.latency.softsegmenter")

INF = float("inf")
PUNCT = set([".", "!", "?", ",", ";", ":", "-", "(", ")"])
CHINESE_PUNCT = set(["。", "！", "？", "，", "；", "：", "—", "（", "）"])
JAPAN_PUNCT = set(["。", "！", "？", "，", "；", "：", "ー", "（", "）"])
ALL_PUNCT = PUNCT.union(CHINESE_PUNCT).union(JAPAN_PUNCT)


class AlignmentOperation:
    """Enum for alignment operations."""
    MATCH = 0
    DELETE = 1
    INSERT = 2
    NONE = 3


@dataclass
class Word:
    """
    Represents a word with associated timing information.

    Attributes:
        text (str): The word text.
        delay (float): The delay timestamp.
        seq_id (Optional[int]): Sequence identifier for alignment.
        elapsed (Optional[float]): Elapsed time (for computation-aware delays).
        main (bool): Whether this is a main word (not a subtoken).
        original (Optional[str]): The original word before tokenization.
        recording_length (Optional[float]): Total recording length.
    """
    text: str
    delay: float
    seq_id: Optional[int] = None
    elapsed: Optional[float] = None
    main: bool = True
    original: Optional[str] = None
    recording_length: Optional[float] = None

    def __repr__(self):
        return (f"Word(text={self.text}, delay={self.delay}, elapsed={self.elapsed}, "
                f"seq_id={self.seq_id}, main={self.main}, original={self.original}, "
                f"recording_length={self.recording_length})")


def unicode_normalize(text: str) -> str:
    """Normalize Unicode text to NFKC form."""
    return unicodedata.normalize("NFKC", text)


def compute_similarity_score(ref_word: Word, hyp_word: Word, char_level: bool) -> float:
    """
    Compute the similarity metric between two words based on Jaccard similarity of
    character sets (or exact match for character-level). Additionally, if one word
    (or character) is punctuation and the other is not, return a negative score to
    discourage alignment of punctuation with non-punctuation.

    Args:
        ref_word (Word): Reference word.
        hyp_word (Word): Hypothesis word.
        char_level (bool): Whether to use character-level comparison.

    Returns:
        float: Similarity score between the words.
    """
    ref_text = ref_word.text
    hyp_text = hyp_word.text

    # If one text is punctuation and the other is not, return a negative score.
    ref_t = ref_text in ALL_PUNCT
    hyp_t = hyp_text in ALL_PUNCT
    if ref_t ^ hyp_t:
        return -INF

    # For character-level, compare lowercased texts directly
    if char_level:
        return float(ref_text == hyp_text)

    ref_set = set(ref_text)
    hyp_set = set(hyp_text)
    inter = len(ref_set & hyp_set)
    union = len(ref_set) + len(hyp_set) - inter

    return (inter / union) if union else 0.0


def _align_sequences(seq1: List[Word], seq2: List[Word], char_level: bool) -> tuple:
    """
    Align two sequences maximizing the similarity metric.

    This function implements a dynamic programming algorithm similar to
    Needleman-Wunsch, but with a custom similarity score and no gap penalties.
    It is also similar to Dynamic Time Warping (DTW) but there is no penalty
    for leading/trailing gaps. The algorithm allows for insertions and
    deletions without penalty, but matches are scored based on the Jaccard
    similarity of character sets (or exact match for character-level).
    The alignment is computed to maximize the total similarity score across
    the aligned sequences.

    Args:
        seq1 (List[Word]): First sequence (typically reference).
        seq2 (List[Word]): Second sequence (typically hypothesis).
        char_level (bool): Whether to use character-level comparison.

    Returns:
        tuple: Two aligned sequences with None for gaps.
    """
    # Initialize the alignment matrix
    n = len(seq1) + 1
    m = len(seq2) + 1
    dp = [[0.0] * m for _ in range(n)]
    dp_back = [[AlignmentOperation.NONE] * m for _ in range(n)]

    # Fill the first row and column of the matrix
    for i in range(n):
        dp[i][0] = 0.0
        dp_back[i][0] = AlignmentOperation.DELETE
    for j in range(m):
        dp[0][j] = 0.0
        dp_back[0][j] = AlignmentOperation.INSERT
    dp[0][0] = 0.0
    dp_back[0][0] = AlignmentOperation.MATCH

    # Fill the alignment matrix
    for i in range(1, n):
        for j in range(1, m):
            match = dp[i - 1][j - 1] + \
                compute_similarity_score(seq1[i - 1], seq2[j - 1], char_level)
            delete = dp[i - 1][j]
            insert = dp[i][j - 1]
            dp[i][j] = max(match, delete, insert)
            if dp[i][j] == match:
                dp_back[i][j] = AlignmentOperation.MATCH
            elif dp[i][j] == delete:
                dp_back[i][j] = AlignmentOperation.DELETE
            else:
                dp_back[i][j] = AlignmentOperation.INSERT

    # Backtrack to find the alignment
    aligned_seq1 = []
    aligned_seq2 = []
    i, j = n - 1, m - 1
    while i > 0 or j > 0:
        if dp_back[i][j] == AlignmentOperation.MATCH:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(seq2[j - 1])
            i -= 1
            j -= 1
        elif dp_back[i][j] == AlignmentOperation.DELETE:
            aligned_seq1.append(seq1[i - 1])
            aligned_seq2.append(None)
            i -= 1
        elif dp_back[i][j] == AlignmentOperation.INSERT:
            aligned_seq1.append(None)
            aligned_seq2.append(seq2[j - 1])
            j -= 1
        else:
            break
    aligned_seq1.reverse()
    aligned_seq2.reverse()
    return aligned_seq1, aligned_seq2


def _process_alignment(
        ref_words: List[Optional[Word]],
        hyp_words: List[Optional[Word]],
        char_level: bool) -> List[Word]:
    """
    Process the alignment to assign sequence IDs to hypothesis words.

    Args:
        ref_words (List[Optional[Word]]): Aligned reference words (with None for gaps).
        hyp_words (List[Optional[Word]]): Aligned hypothesis words (with None for gaps).
        char_level (bool): Whether using character-level alignment.

    Returns:
        List[Word]: Processed hypothesis words with assigned sequence IDs.
    """
    assert len(ref_words) == len(hyp_words), \
        "Number of reference and hypothesis words do not match."

    def get_next_non_none_ref(i):
        while i < len(ref_words) and ref_words[i] is None:
            i += 1
        if i == len(ref_words):
            return None
        return i, ref_words[i]

    new_hyp_words = []
    last_ref = None
    nexti = 0
    for i, (ref, hyp) in enumerate(zip(ref_words, hyp_words)):
        if ref is None and i >= nexti:
            if hyp is not None:
                next_ref_info = get_next_non_none_ref(i)
                if next_ref_info is not None:
                    nexti, next_ref = next_ref_info
                    if next_ref is not None and hyp.delay >= next_ref.delay:
                        last_ref = next_ref

        # last_ref can be set to next_ref to avoid non-monotonicity
        if ref is not None and i >= nexti:
            last_ref = ref
        if hyp is not None:
            if ref is None:
                hyp.seq_id = last_ref.seq_id if last_ref is not None else 0
            else:
                hyp.seq_id = ref.seq_id
            new_hyp_words.append(hyp)

    return new_hyp_words


def _process_sample_alignment(
        args: Tuple[int, List[Word], List[Word], bool]
) -> Tuple[int, List[Word]]:
    """
    Process alignment for a single sample. Top-level function for multiprocessing.

    Args:
        args: Tuple of (sample_index, ref_words, hyp_words, char_level).

    Returns:
        Tuple of (sample_index, processed_hyp_words).
    """
    i, ref_words, hyp_words, char_level = args
    aligned_ref, aligned_hyp = _align_sequences(ref_words, hyp_words, char_level)
    processed_hyp = _process_alignment(aligned_ref, aligned_hyp, char_level)
    return i, processed_hyp


class SoftSegmenterBasedLatencyScorer(SegmenterBasedScorer):
    """
    Abstract base class for scorers that require aligned system outputs and references through
    SoftSegmenter alignment.

    This class wraps a latency scorer and applies the SoftSegmenter alignment algorithm from
    `"Better Late Than Never: Evaluation of Latency Metrics for Simultaneous Speech-to-Text
    Translation" <https://arxiv.org/abs/2509.17349>`_ to hypotheses before scoring.

    Subclasses must implement :meth:`_do_score`, which operates on
    :class:`ResegmentedLatencyScoringSample` instances where hypotheses and references are aligned.

    Args:
        args: Parsed arguments containing latency_unit and optionally moses_lang and num_workers.

    Example:
        >>> class CustomLatencyScorer(SoftSegmenterBasedLatencyScorer):
        ...     def _do_score(self, samples):
        ...         # Compute a custom latency score
        ...         return LatencyScores(...)
    """
    def __init__(self, args):
        super().__init__(args)
        self.latency_unit = args.latency_unit
        self.moses_lang = getattr(args, 'moses_lang', None)
        self.num_workers = getattr(args, 'num_workers', None)

    def requires_reference(self) -> bool:
        return True

    @abstractmethod
    def _do_score(self, samples: List[ResegmentedLatencyScoringSample]) -> LatencyScores:
        """
        Compute latency scores on resegmented samples.

        Subclasses must override this method.

        Args:
            samples (List[ResegmentedLatencyScoringSample]): Aligned
                hypothesis–reference pairs with delay information.

        Returns:
            LatencyScores: The computed latency metrics.
        """
        ...

    def _create_words_from_output(
            self,
            output: OutputWithDelays,
            char_level: bool,
            recording_length: float) -> List[Word]:
        """
        Convert OutputWithDelays to a list of Word objects.

        Args:
            output (OutputWithDelays): The output with delays.
            char_level (bool): Whether to use character-level units.
            recording_length (float): Total length of the recording.

        Returns:
            List[Word]: List of Word objects.
        """
        units = text_items(output.final_text, "char" if char_level else "word")
        assert len(units) == len(output.ideal_delays), \
            f"Number of units ({len(units)}) and delays ({len(output.ideal_delays)}) do not match"
        lu = len(units)
        l_ca = len(output.computational_aware_delays)
        assert lu == l_ca, f"Number of units ({lu}) and CA delays ({l_ca}) do not match"

        words = []
        for unit, delay, elapsed in zip(
            units, output.ideal_delays, output.computational_aware_delays):
            words.append(Word(
                text=unit,
                delay=delay,
                elapsed=elapsed,
                recording_length=recording_length
            ))
        return words

    def _create_words_from_references(
            self,
            references: List[ReferenceSentenceDefinition],
            char_level: bool) -> List[Word]:
        """
        Convert reference sentences to a list of Word objects.

        Args:
            references (List[ReferenceSentenceDefinition]): Reference sentences.
            char_level (bool): Whether to use character-level units.

        Returns:
            List[Word]: List of Word objects with sequence IDs.
        """
        words = []
        for i, ref in enumerate(references):
            ref_text = unicode_normalize(ref.content).lower()
            units = text_items(ref_text, "char" if char_level else "word")
            delay = ref.start_time
            for unit in units:
                words.append(Word(text=unit, delay=delay, seq_id=i))
        return words

    def tokenize_words(self, words: List[Word], tokenizer: Callable) -> List[Word]:
        """
        Tokenize words using Moses tokenizer.

        Args:
            words (List[Word]): List of words to tokenize.
            tokenizer (callable): Callable that tokenizes a string.

        Returns:
            List[Word]: Tokenized words with subtokens marked.
        """

        tokenized_words: List[Word] = []
        for word in words:
            text = unicode_normalize(word.text).lower()
            result = tokenizer(text)  # type: ignore
            # Ensure result is a list
            if isinstance(result, str):
                tokens: List[str] = [result]
            else:
                tokens = list(result) if result else [text]  # type: ignore
            main = True
            for token in tokens:
                tokenized_words.append(Word(
                    text=token,
                    delay=word.delay,
                    seq_id=word.seq_id,
                    elapsed=word.elapsed,
                    main=main,
                    original=word.text if main else None,
                    recording_length=word.recording_length
                ))
                main = False
        return tokenized_words

    def score(self, samples: List[LatencyScoringSample]) -> LatencyScores:
        char_level = self.latency_unit == "char"

        # Prepare alignment arguments for all samples
        alignment_args = []
        sample_metadata = []  # Store per-sample metadata for post-processing

        if self.moses_lang is None:
            LOGGER.warning(
                "moses_lang not specified; defaulting to character-level tokenization. "
                "This is recommended for Chinese/Japanese, but for other languages it is "
                "recommended to specify a moses_lang for proper tokenization. Set "
                "--moses-lang to the appropriate language code (e.g., 'en', 'de') "
                "or to 'zh'/'ja' to skip tokenization.")
            tokenizer = lambda x: x  # Identity tokenizer for character-level or when moses_lang is None
        else:
            tokenizer = MosesTokenizer(lang=self.moses_lang, no_escape=True)

        for idx, sample in enumerate(samples):
            assert sample.reference is not None, \
                "Cannot realign hypothesis to missing reference"

            # Calculate total recording length
            recording_length = max(
                ref.start_time + ref.duration for ref in sample.reference
            )

            # Convert references to Word objects
            ref_words = self._create_words_from_references(sample.reference, char_level)
            ref_words = self.tokenize_words(ref_words, tokenizer)

            # Convert hypothesis to Word objects
            hyp_words = self._create_words_from_output(
                sample.hypothesis, char_level, recording_length)
            hyp_words = self.tokenize_words(hyp_words, tokenizer)

            alignment_args.append((idx, ref_words, hyp_words, char_level))
            sample_metadata.append(sample)

        # Parallelize alignment computation using multiprocessing Pool
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(_process_sample_alignment, alignment_args)

        # Sort results by sample index to maintain original order
        results.sort(key=lambda x: x[0])

        # Post-process alignment results into resegmented samples
        resegmented_samples = []
        for sample_idx, processed_hyp in results:
            sample = sample_metadata[sample_idx]

            # Group hypothesis words by sequence ID
            new_segmentation = {}
            for i in range(len(sample.reference)):
                new_segmentation[i] = []

            for word in processed_hyp:
                if word.main and word.seq_id is not None:
                    new_segmentation[word.seq_id].append(word)

            # Create OutputWithDelays for each segment
            resegmented_hypos_with_delays = []
            for i, ref in enumerate(sample.reference):
                segment_words = new_segmentation.get(i, [])
                if len(segment_words) == 0:
                    # Empty segment
                    resegmented_hypos_with_delays.append(
                        OutputWithDelays("", [], [])
                    )
                else:
                    # Reconstruct text from main words
                    text_parts = [w.original if w.original else w.text for w in segment_words]
                    if char_level:
                        text = "".join(text_parts)
                    else:
                        text = " ".join(text_parts)

                    # Offset delays relative to segment start
                    ideal_delays = [w.delay - ref.start_time for w in segment_words]
                    ca_delays = [w.elapsed - ref.start_time for w in segment_words]

                    resegmented_hypos_with_delays.append(
                        OutputWithDelays(text, ideal_delays, ca_delays)
                    )

            resegmented_samples.append(ResegmentedLatencyScoringSample(
                sample.audio_name,
                resegmented_hypos_with_delays,
                sample.reference,
            ))

        return self._do_score(resegmented_samples)

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--moses-lang",
            type=str,
            default=None,
            help='Language code for Moses tokenizer (e.g., "en", "de"). '
            "Use None for Chinese/Japanese or to skip tokenization.",
        )
        parser.add_argument(
            "--num-workers",
            type=int,
            default=None,
            help="Number of worker processes for parallel alignment. "
            "Defaults to the number of CPU cores if not specified."
        )