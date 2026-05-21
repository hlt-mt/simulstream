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
import transformers

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


class UltravoxDOA(DecoderOnlyAttention):
    """
    Decoder-Only Attention agent for UltraVox.

    Architecture: Whisper-large-v3-turbo encoder → stack_factor=8 projector → Llama-3.1-8B.
    Audio is injected into the LLM embeddings at ``<|audio|>`` placeholder positions.
    The pipeline preprocessor handles tokenization and audio feature extraction.

    Extra config fields
    -------------------
    hf_model_name : str
        Default: ``"fixie-ai/ultravox-v0_6-llama-3_1-8b"``.
    repetition_penalty : float
        Default: ``1.0``.
    no_repeat_ngram_size : int
        Default: ``0``.
    """

    BOW_PREFIX = " "
    # Whisper-large-v3-turbo: 50 frames/s; stack_factor=8 → 50/8 tokens/s
    # stride = 16000 / (50/8) = 2560 samples per audio token
    AUDIO_TOKEN_STRIDE = 2560

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
            getattr(config, "model_path", "fixie-ai/ultravox-v0_6-llama-3_1-8b"),
        )
        attn_impl = getattr(config, "attn_implementation", "eager")

        cls.pipe = transformers.pipeline(
            model=model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            model_kwargs={"attn_implementation": attn_impl},
        )
        cls.model = cls.pipe.model
        cls.model.eval()

    def build_prompt(self) -> str:
        return (
            TEMPLATED_SPEECH_PROMPT
            .replace("{src_lang}", LANG_MAPPER.get(self.src_lang, self.src_lang))
            .replace("{tgt_lang}", LANG_MAPPER.get(self.tgt_lang, self.tgt_lang))
        )

    def build_processor_inputs(self, waveform: np.ndarray) -> dict:
        prefix = self.build_raw_text_prefix()
        turns = [
            {
                "role": "system",
                "content": self.build_prompt(),
            },
        ]
        # The pipeline preprocessor tokenizes the conversation, extracts audio
        # features, and returns audio_token_start_idx + audio_token_len alongside
        # the standard input_ids/attention_mask/audio_values tensors.
        inputs = self.pipe.preprocess(
            {"audio": waveform, "turns": turns, "sampling_rate": SAMPLE_RATE}
        )

        if prefix:
            prefix_ids = self.pipe.tokenizer(
                prefix,
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids
            inputs["input_ids"] = torch.cat([inputs["input_ids"], prefix_ids], dim=1)
            inputs["attention_mask"] = torch.cat(
                [inputs["attention_mask"], torch.ones_like(prefix_ids)], dim=1
            )

        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

    def _find_audio_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Ultravox pipeline provides audio_token_start_idx and audio_token_len
        # directly — no need to scan input_ids for a placeholder token.
        # Called with the full inputs dict via _generate.
        raise NotImplementedError("Use _find_audio_positions_from_inputs instead.")

    def _find_audio_positions_from_inputs(self, inputs: dict) -> torch.Tensor:
        start = inputs["audio_token_start_idx"][0].item()
        length = inputs["audio_token_len"][0].item()
        return torch.arange(start, start + length, device=self.device)

    def _generate(self, inputs: dict) -> Tuple[List[str], torch.Tensor]:
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        audio_positions = self._find_audio_positions_from_inputs(inputs)
        audio_len = audio_positions.shape[0]

        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            repetition_penalty=self.repetition_penalty,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            output_attentions=True,
            return_dict_in_generate=True,
            do_sample=False,
            temperature=self.temperature,
            eos_token_id=[
                self.pipe.tokenizer.eos_token_id,
                self.pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            ],
        )

        new_ids = output.sequences[:, input_len:]
        new_tokens = [
            self.pipe.tokenizer.decode([token_id], skip_special_tokens=True)
            for token_id in new_ids[0]
        ]

        prefill_attn = self.mean_attn_over_heads_and_selected_layers(output.attentions[0])
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