
from typing import List

from simulstream.metrics.readers import text_items
from simulstream.metrics.scorers.latency import LatencyScorer


class SegmenterBasedScorer(LatencyScorer):

    def __init__(self, args):
        super().__init__(args)
        self.latency_unit = args.latency_unit

    def _split_delays_by_segmented_text(
            self, delays: List[float], segmented_text: List[str]) -> List[List[float]]:
        """
        Assign delay values to the corresponding segmented hypotheses.

        Args:
            delays (List[float]): Delay values (per token or per char).
            segmented_text (List[str]): Segmented hypothesis strings.

        Returns:
            List[List[float]]: Delays split per segment.
        """
        segmented_delays = []
        index = 0

        for segment in segmented_text:
            segment_len = len(text_items(segment, self.latency_unit))
            segmented_delays.append(delays[index:index + segment_len])
            index += segment_len
        assert len(delays) == index, \
            f"Index {index} should have reached end of delays ({len(delays)})"
        return segmented_delays
