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
import sys
from typing import List

from simulstream.metrics.scorers.quality import register_quality_scorer
from simulstream.metrics.scorers.quality.mwersegmenter import MWERSegmenterBasedQualityScorer, \
    ResegmentedQualityScoringSample

try:
    from comet import download_model, load_from_checkpoint
except ImportError:
    sys.exit("Please install comet first with `pip install unbabel-comet`.")


@register_quality_scorer("comet")
class CometScorer(MWERSegmenterBasedQualityScorer):
    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        self.batch_size = args.batch_size
        model_path = download_model(args.model)
        self.model = load_from_checkpoint(model_path)
        self.model.eval()

    def _do_score(self, samples: List[ResegmentedQualityScoringSample]) -> float:
        comet_data = []
        for sample in samples:
            for hyp, ref, src in zip(sample.hypothesis, sample.reference, sample.source):
                comet_data.append({
                    "src": src.strip(),
                    "mt": hyp.strip(),
                    "ref": ref.strip()
                })

        metric = self.model.predict(comet_data, batch_size=self.batch_size)
        return metric.system_score

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--model", type=str, default="Unbabel/wmt22-comet-da")
        parser.add_argument("--batch-size", type=int, default=16)

    def requires_source(self) -> bool:
        return True
