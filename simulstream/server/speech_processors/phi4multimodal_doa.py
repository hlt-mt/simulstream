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

import torch
import numpy as np

from types import SimpleNamespace
from typing import List, Tuple

from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

from simulstream.server.speech_processors import SAMPLE_RATE, class_load
from simulstream.server.speech_processors.base_doa import DecoderOnlyAttention, TEMPLATED_SPEECH_PROMPT, LANG_MAPPER


class Phi4MultimodalDOA(DecoderOnlyAttention):
    """
    Decoder-Only Attention agent for ``microsoft/Phi-4-multimodal-instruct``.

    Extra config fields
    -------------------
    model_path : str
        Default: ``"microsoft/Phi-4-multimodal-instruct"``
    target_lang : str
        Target language when ``task="translate"``.  Default: ``"English"``
    """

    # Phi-4 special tokens
    _USER_START  = "<|user|>"
    _AUDIO_TOKEN = "<|audio_1|>"
    _END_TOKEN   = "<|end|>"
    _ASST_START  = "<|assistant|>"

    BOW_PREFIX = " "
    ENCODER_SUBSAMPLING_FACTOR = 8
    HOP_LENGTH = 160    # 10ms at 16kHz

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.bow_prefix = self.BOW_PREFIX
        text_history_cls = class_load(self.text_history_config.type)
        self.text_history_method = text_history_cls(self.text_history_config, self.bow_prefix)
        self.audio_subsampling_factor = self.ENCODER_SUBSAMPLING_FACTOR * self.HOP_LENGTH

    @classmethod
    def load_model(cls, config: SimpleNamespace) -> None:
        model_path = "microsoft/Phi-4-multimodal-instruct"

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

    @property
    def audio_max_len(self) -> int:
        return getattr(self.config, "audio_max_frames", 480_000)

    def build_prompt(self) -> str:
        filled_prompt = (
            TEMPLATED_SPEECH_PROMPT
            .replace("{src_lang}", LANG_MAPPER[self.src_lang])
            .replace("{tgt_lang}", LANG_MAPPER[self.tgt_lang]))
        prefix = self.text_history if self.text_history else ""
        return (
            f"{self._USER_START}{self._AUDIO_TOKEN}"
            f"{filled_prompt}{self._END_TOKEN}"
            f"{self._ASST_START}{prefix}"
        )

    def build_processor_inputs(self, waveform: np.ndarray) -> dict:
        return self.processor(
            text=self.build_prompt(),
            audios=[(waveform, SAMPLE_RATE)],
            return_tensors="pt",
        ).to(self.device)

    def _generate(self, inputs: dict) -> Tuple[List[str], torch.Tensor]:
        """
        Run greedy generation and build the proxy cross-attention matrix.

        ``output.attentions`` (use_cache=True) contains the self-attention scores,
        for each step and layer. H is the dimension of the attention heads.
        ───────────────────────────────────────────────────────────────────────────────────────────
        output.attentions[0][layer]  → (1, H, input_len, input_len)  ← prefill
        output.attentions[i][layer]  → (1, H, 1, input_len+i)        ← new token i

        Returns
        -------
        List[str]
            A list of the newly generated tokens (n_new).
        torch.Tensor
            Proxy cross-attention scores extracted from the self-attention scores,
            averaged over heads at ``self.cross_attn_layer`` (prefix + n_new, audio_len).
        """
        input_ids = inputs["input_ids"]  # (1, input_len)
        input_len = input_ids.shape[1]

        # Locate audio positions ──────────────────────────────────────────────────────────────────
        AUDIO_SPECIAL_TOKEN_ID = 200011  # _AUDIO_SPECIAL_TOKEN_ID in modeling_phi4mm.py
        audio_positions = (input_ids[0] == AUDIO_SPECIAL_TOKEN_ID).nonzero(as_tuple=True)[0]
        audio_len = audio_positions.shape[0]

        # Generate ────────────────────────────────────────────────────────────────────────────────
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            generation_config=self.generation_config,
            num_logits_to_keep=1,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,
        )

        # Decode newly generated tokens only ──────────────────────────────────────────────────────
        new_ids = output.sequences[:, input_len:]  # (1, n_new)
        new_tokens = [
            self.processor.tokenizer.decode([t], skip_special_tokens=True)
            for t in new_ids[0]
        ]

        # Build proxy cross-attention for the hypothesis (prefix + new_tokens) ────────────────────
        # Prefix rows from the prefill pass
        # output.attentions[0][layer]: (1, H, input_len, input_len)
        prefill_attn = (output.attentions[0][self.cross_attn_layer][0]
                        .mean(dim=0))  # (input_len, input_len)
        prefix_rows = prefill_attn[
            self.cross_attn_layer:, :][:, audio_positions]  # (n_prefix, audio_len)
        # New-token rows: one per step, each (1, H, 1, input_len+i)
        new_rows = [
            step_attn[self.cross_attn_layer][0]
            .mean(dim=0).squeeze(0)[audio_positions]  # (audio_len,)
            for step_attn in output.attentions[1:-1]    # avoid attention of <|end|> token
        ]
        new_attn = torch.stack(new_rows, dim=0) if new_rows else \
            torch.zeros(0, max(audio_len, 1), device=self.device)

        cross_attn = torch.cat([prefix_rows, new_attn], dim=0) # (n_prefix + n_new, audio_len)

        return new_tokens, cross_attn


    def tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens).strip()
