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

from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

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


class Qwen2_5OmniDOA(DecoderOnlyAttention):
    """
    Decoder-Only Attention agent for ``Qwen/Qwen2.5-Omni-*``.

    Extra config fields
    -------------------
    hf_model_name : str
        Default: ``"Qwen/Qwen2.5-Omni-7B"``.
        ``"Qwen/Qwen2.5-Omni-3B"`` is also supported.
    repetition_penalty : float
        Repetition penalty for text generation. Default: ``1.0``.
    no_repeat_ngram_size : int
        N-gram blocking size for text generation. Default: ``0``.
    """

    BOW_PREFIX = " "
    AUDIO_TOKEN_STRIDE = 640
    AUDIO_TOKEN_INDEX = 151646
    AUDIO_START_TOKEN_ID = 151647
    AUDIO_END_TOKEN_ID = 151648
    SYSTEM_PROMPT = (
        "You are a speech translation system. "
        "Translate the audio input into the target language. "
        "Output only the translation. "
        "Do not ask questions, do not add commentary, do not simulate a conversation, "
        "do not write 'Human:', 'Assistant:', or any dialogue markers, including newlines. "
        "If the audio is unclear or incomplete, output only what you can translate and stop."
    )

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.bow_prefix = self.BOW_PREFIX
        text_history_cls = class_load(self.text_history_config.type)
        self.text_history_method = text_history_cls(self.text_history_config, self.bow_prefix)
        self.audio_subsampling_factor = self.AUDIO_TOKEN_STRIDE
        self.use_video =  getattr(self.config, "use_video", False)
        self.repetition_penalty = getattr(self.config, "repetition_penalty", 1.05)
        self.temperature = getattr(self.config, "temperature", 1.0)
        self.no_repeat_ngram_size = getattr(self.config, "no_repeat_ngram_size", 5)

    @classmethod
    def load_model(cls, config: SimpleNamespace) -> None:
        model_name = getattr(
            config,
            "hf_model_name",
            getattr(config, "model_path", "Qwen/Qwen2.5-Omni-7B"),
        )
        attn_impl = getattr(config, "attn_implementation", "eager") #"flash_attention_2")

        cls.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=attn_impl,
        )
        cls.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
        cls.model.eval()

    def build_prompt(self) -> str:
        return f"Translate the audio to {LANG_MAPPER[self.tgt_lang]}."

    def build_processor_inputs(self, waveform: np.ndarray) -> dict:
        prompt_text = self.build_prompt()

        conversation = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": waveform},
                    {"type": "text", "text": prompt_text},
                ],
            },
        ]

        prompt = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        prefix = self.build_raw_text_prefix()

        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

        return self.processor(
            text=f"{prompt}{prefix}",
            audio=audios,
            images=images,
            videos=videos,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        ).to(self.device)

    def _find_audio_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        audio_positions = (input_ids[0] == self.AUDIO_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        if audio_positions.numel() > 0:
            return audio_positions

        start_positions = (input_ids[0] == self.AUDIO_START_TOKEN_ID).nonzero(as_tuple=True)[0]
        end_positions = (input_ids[0] == self.AUDIO_END_TOKEN_ID).nonzero(as_tuple=True)[0]
        if start_positions.numel() == 0 or end_positions.numel() == 0:
            raise ValueError(
                "Qwen2.5-Omni audio tokens were not found in the prompt. Checked "
                "`audio_token_index`, `<|audio_bos|>`, and `<|audio_eos|>`."
            )

        start_pos = start_positions[0]
        end_positions = end_positions[end_positions > start_pos]
        if end_positions.numel() == 0:
            raise ValueError("Qwen2.5-Omni found `<|audio_bos|>` but not a matching `<|audio_eos|>`.")

        end_pos = end_positions[0]
        if end_pos <= start_pos + 1:
            raise ValueError("Qwen2.5-Omni found empty audio span between `<|audio_bos|>` and `<|audio_eos|>`.")

        return torch.arange(start_pos + 1, end_pos, device=input_ids.device)

    def _generate(self, inputs: dict) -> Tuple[List[str], torch.Tensor]:
        input_ids = inputs["input_ids"]
        input_len = input_ids.shape[1]

        audio_positions = self._find_audio_positions(input_ids)
        audio_len = audio_positions.shape[0]

        output = self.model.generate(
            **inputs,
            use_audio_in_video=True,
            return_audio=False,
            thinker_max_new_tokens=self.max_new_tokens,
            thinker_repetition_penalty=self.repetition_penalty,
            thinker_no_repeat_ngram_size=self.no_repeat_ngram_size,
            thinker_output_attentions=True,
            thinker_return_dict_in_generate=True,
            thinker_do_sample=False,
            temperature=self.temperature,
            eos_token_id=[151643, 151645],  # <|endoftext|> and <|im_end|>
        )
        if isinstance(output, tuple):
            output = output[0]

        new_ids = output.sequences[:, input_len:]
        stop_ids = {151643, 151645}
        new_tokens = []
        for token_id in new_ids[0]:
            if token_id.item() in stop_ids:
                break
            new_tokens.append(
                self.processor.tokenizer.decode([token_id], skip_special_tokens=True)
            )

        prefill_attn = self.mean_attn_over_heads_and_selected_layers(output.attentions[0])
        prefix_len = len(self.text_history) if self.text_history else 0
        if prefix_len > 0:
            prefix_rows = prefill_attn[input_len - prefix_len:, :][:, audio_positions]
        else:
            prefix_rows = torch.zeros(0, max(audio_len, 1), device=self.device)

        first_new_row = prefill_attn[-1:, audio_positions] if new_tokens else \
            torch.zeros(0, max(audio_len, 1), device=self.device)
        new_rows = [
            self.mean_attn_over_heads_and_selected_layers(step_attn).squeeze(0)[audio_positions]
            for step_attn in output.attentions[1:]
        ]
        subsequent_new_attn = torch.stack(new_rows, dim=0) if new_rows else \
            torch.zeros(0, max(audio_len, 1), device=self.device)
        new_attn = torch.cat([first_new_row, subsequent_new_attn], dim=0)

        cross_attn = torch.cat([prefix_rows, new_attn], dim=0)
        cross_attn = self.normalize_attn(cross_attn)
        return new_tokens, cross_attn

    def tokens_to_string(self, tokens: List[str]) -> str:
        return "".join(tokens)
