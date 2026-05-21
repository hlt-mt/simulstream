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

        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": "placeholder"},
                    {"type": "text", "text": self.build_prompt()},
                ],
            },
        ]

        text = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )

        inputs = self.processor(
            text=f"{text}{prefix}",
            audio=[waveform],
            return_tensors="pt",
            padding=True,
        )
        return inputs.to(self.device)

    def _find_audio_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids[0] == self.AUDIO_TOKEN_INDEX).nonzero(as_tuple=True)[0]

    def _generate(self, inputs: dict) -> Tuple[List[str], torch.Tensor]:
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        audio_positions = self._find_audio_positions(input_ids)
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
            eos_token_id=[151643, 151645],  # <|endoftext|> and <|im_end|>
        )

        new_ids = output.sequences[:, input_len:]
        new_tokens = [
            self.processor.tokenizer.decode([token_id], skip_special_tokens=True)
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
