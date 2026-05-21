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

import logging
from types import SimpleNamespace
from typing import List, Tuple

import numpy as np
import torch

from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

from simulstream.server.speech_processors import SAMPLE_RATE, class_load
from simulstream.server.speech_processors.base_doa import (
    DecoderOnlyAttention,
    LANG_MAPPER,
    TEMPLATED_SPEECH_PROMPT,
)

from transformers import set_seed
torch.manual_seed(42)
set_seed(42)


logger = logging.getLogger(__name__)


class Qwen2AudioDOA(DecoderOnlyAttention):
    """
    Decoder-Only Attention agent for ``Qwen/Qwen2-Audio-7B-Instruct``.

    Architecture: Whisper encoder (subsampling factor 2) → linear projector → Qwen2-7B.
    Audio is serialized through the official Qwen2-Audio chat template and then
    expanded by the processor into repeated ``<|AUDIO|>`` placeholder tokens in
    ``input_ids``.

    Extra config fields
    -------------------
    hf_model_name : str
        Default: ``"Qwen/Qwen2-Audio-7B-Instruct"``.
    repetition_penalty : float
        Default: ``1.0``.
    no_repeat_ngram_size : int
        Default: ``0``.
    """

    BOW_PREFIX = " "
    # Whisper encoder: 50 frames/s, subsampling factor 2 → 25 tokens/s
    # stride = 16000 / 25 = 640 samples per audio token
    AUDIO_TOKEN_STRIDE = 640
    AUDIO_TOKEN_INDEX = 151646  # <|AUDIO|>

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.bow_prefix = self.BOW_PREFIX
        text_history_cls = class_load(self.text_history_config.type)
        self.text_history_method = text_history_cls(self.text_history_config, self.bow_prefix)
        self.audio_subsampling_factor = self.AUDIO_TOKEN_STRIDE
        self.repetition_penalty = getattr(self.config, "repetition_penalty", 1.0)
        self.temperature = getattr(self.config, "temperature", 1.0)
        self.no_repeat_ngram_size = getattr(self.config, "no_repeat_ngram_size", 0)

    @classmethod
    def load_model(cls, config: SimpleNamespace) -> None:
        model_name = getattr(
            config,
            "hf_model_name",
            getattr(config, "model_path", "Qwen/Qwen2-Audio-7B-Instruct"),
        )
        attn_impl = getattr(config, "attn_implementation", "eager")

        cls.processor = AutoProcessor.from_pretrained(model_name)
        cls.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=attn_impl,
        )
        cls.model.eval()

    def build_prompt(self) -> str:
        return (
            TEMPLATED_SPEECH_PROMPT
            .replace("{src_lang}", LANG_MAPPER.get(self.src_lang, self.src_lang))
            .replace("{tgt_lang}", LANG_MAPPER.get(self.tgt_lang, self.tgt_lang))
        )

    def build_processor_inputs(self, waveform: np.ndarray) -> dict:
        prefix = self.build_raw_text_prefix()
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            "Audio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
            f"{self.build_prompt()}<|im_end|>\n"
            f"<|im_start|>assistant\n{prefix}"
        )

        inputs = self.processor(
            text=prompt,
            audios=[waveform],
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        return inputs.to(self.device)

    def _find_audio_positions(self, inputs: dict, prefill_len: int) -> torch.Tensor:
        input_ids = inputs["input_ids"]
        raw_audio_positions = (input_ids[0] == self.AUDIO_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        if raw_audio_positions.numel() == 0:
            raise ValueError("Qwen2-Audio audio placeholder token was not found in input_ids.")

        feature_attention_mask = inputs.get("feature_attention_mask")
        if feature_attention_mask is not None:
            audio_feat_lengths, audio_output_lengths = self.model.audio_tower._get_feat_extract_output_lengths(
                feature_attention_mask.sum(-1)
            )
            valid_audio_len = int(audio_output_lengths[0].item())
            if 0 < valid_audio_len <= raw_audio_positions.numel():
                return raw_audio_positions[:valid_audio_len]

        if raw_audio_positions.numel() > 1 or prefill_len == input_ids.shape[1]:
            return raw_audio_positions

        expanded_audio_len = prefill_len - input_ids.shape[1] + 1
        if expanded_audio_len <= 0:
            raise ValueError(
                "Qwen2-Audio audio expansion length is invalid. "
                f"prefill_len={prefill_len}, input_len={input_ids.shape[1]}."
            )

        start = raw_audio_positions[0].item()
        return torch.arange(start, start + expanded_audio_len, device=input_ids.device)

    def _generate(self, inputs: dict) -> Tuple[List[str], torch.Tensor]:
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,
            temperature=self.temperature,
        )

        new_ids = output.sequences[:, input_len:]
        new_tokens = [
            self.processor.tokenizer.decode([token_id], skip_special_tokens=True)
            for token_id in new_ids[0]
        ]

        prefill_attn = self.mean_attn_over_heads_and_selected_layers(output.attentions[0])
        audio_positions = self._find_audio_positions(inputs, prefill_attn.shape[0])
        audio_len = audio_positions.shape[0]
        prefix_len = len(self.text_history) if self.text_history else 0
        empty_attn = torch.zeros(0, audio_len, device=self.device)

        prefix_rows = prefill_attn[input_len - prefix_len:, :][:, audio_positions] \
            if prefix_len > 0 else empty_attn
        first_new_row = prefill_attn[-1:, audio_positions] if new_tokens else empty_attn
        new_rows = [
            self.mean_attn_over_heads_and_selected_layers(step_attn).squeeze(0)[audio_positions]
            for step_attn in output.attentions[1:]
        ]
        subsequent_new_attn = torch.stack(new_rows, dim=0) if new_rows else empty_attn
        new_attn = torch.cat([first_new_row, subsequent_new_attn], dim=0)

        cross_attn = torch.cat([prefix_rows, new_attn], dim=0)
        cross_attn = self.normalize_attn(cross_attn)
        return new_tokens, cross_attn

    def tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)
