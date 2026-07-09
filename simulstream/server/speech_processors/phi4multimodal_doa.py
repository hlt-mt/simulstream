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

from types import SimpleNamespace
from typing import List, Any, Dict, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.base_doa import DecoderOnlyAttention, get_language_name


class Phi4MultimodalDOA(DecoderOnlyAttention):
    """
    Decoder-Only Attention agent for Phi4-Multimodal.
    """

    # Phi-4 special tokens
    _USER_START = "<|user|>"
    _AUDIO_TOKEN = "<|audio_1|>"
    _END_TOKEN = "<|end|>"
    _ASST_START = "<|assistant|>"

    ENCODER_SUBSAMPLING_FACTOR = 8
    HOP_LENGTH = 160  # 10ms at 16kHz
    AUDIO_SPECIAL_TOKEN_ID = 200011  # _AUDIO_SPECIAL_TOKEN_ID in modeling_phi4mm.py

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.audio_subsampling_factor = self.ENCODER_SUBSAMPLING_FACTOR * self.HOP_LENGTH

    @classmethod
    def load_model(cls, config: SimpleNamespace) -> None:
        model_path = getattr(
            config,
            "hf_model_name",
            getattr(config, "model_path", "microsoft/Phi-4-multimodal-instruct"),
        )

        cls.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        cls.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation="eager",
        )
        cls.model.eval()
        cls.generation_config = GenerationConfig.from_pretrained(model_path)

    def build_prompt(self) -> str:
        filled_prompt = self.prompt.replace("{tgt_lang}", get_language_name(self.tgt_lang))
        raw_prefix = self.build_raw_text_prefix()
        prompt = (
            f"{self._USER_START}{self._AUDIO_TOKEN}"
            f"{filled_prompt}{self._END_TOKEN}"
            f"{self._ASST_START}{raw_prefix}"
        )
        return prompt

    def build_processor_inputs(self, waveform: np.ndarray) -> dict:
        return self.processor(
            text=self.build_prompt(),
            audios=[(waveform, SAMPLE_RATE)],
            return_tensors="pt",
        )

    def _do_generate(self, inputs: Dict[str, Any]) -> Tuple[List[str], List[torch.Tensor]]:
        input_len = inputs["input_ids"].shape[1]
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            generation_config=self.generation_config,
            num_logits_to_keep=1,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,
        )
        new_tokens = self.processor.tokenizer.convert_ids_to_tokens(
            output.sequences[0, input_len:], skip_special_tokens=True)
        return new_tokens, output.attentions

    def _find_audio_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids[0] == self.AUDIO_SPECIAL_TOKEN_ID).nonzero(as_tuple=True)[0]
