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

from types import SimpleNamespace
from typing import Callable, Dict, List


def build_hf_detokenizer(config: SimpleNamespace) -> Callable[[List[str]], str]:
    from transformers import AutoProcessor

    assert hasattr(config, "hf_model_name"), \
        "`hf_model_name` required in the eval config for `hf` detokenizer"
    processor = AutoProcessor.from_pretrained(config.hf_model_name)

    def detokenize(input_tokens: List[str]) -> str:
        return processor.tokenizer.convert_tokens_to_string(''.join(input_tokens))

    return detokenize


def build_canary_detokenizer(config: SimpleNamespace) -> Callable[[List[str]], str]:
    from nemo.collections.asr.models import ASRModel

    assert hasattr(config, "model_name"), \
        "`model_name` required in the eval config for `canary` detokenizer"
    tokenizer = ASRModel.from_pretrained(model_name=config.model_name).tokenizer

    def detokenize(input_tokens: List[str]) -> str:
        return tokenizer.tokens_to_text(input_tokens)

    return detokenize


def build_simuleval_detokenizer(config: SimpleNamespace) -> Callable[[List[str]], str]:
    """ SimulEval detokenizer from https://github.com/facebookresearch/SimulEval/blob/
    536de8253b82d805c9845440169a5010ff507357/simuleval/evaluator/instance.py#L233"""
    if config.latency_unit == "word":
        def detokenize(input_tokens: List[str]) -> str:
            return " ".join(input_tokens)
    elif config.latency_unit == "char":
        def detokenize(input_tokens: List[str]) -> str:
            return "".join(input_tokens)
    elif config.latency_unit == "spm":
        def detokenize(input_tokens: List[str]) -> str:
            return "".join(input_tokens).replace("▁", " ").strip()
    else:
        raise NotImplementedError

    return detokenize


_DETOKENIZER_BUILDER_MAP: Dict[str, Callable[[SimpleNamespace], Callable[[List[str]], str]]] = {
    "hf": build_hf_detokenizer,
    "canary": build_canary_detokenizer,
    "simuleval": build_simuleval_detokenizer
}


def get_detokenizer(config: SimpleNamespace) -> Callable[[List[str]], str]:
    return _DETOKENIZER_BUILDER_MAP[config.detokenizer_type](config)
