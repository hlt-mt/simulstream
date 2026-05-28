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
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

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


class Qwen3OmniDOA(DecoderOnlyAttention):
    """
    Decoder-Only Attention agent for Qwen3-Omni.

    Extra config fields
    -------------------
    repetition_penalty : float
        Repetition penalty for text generation. Default: ``1.05``.
    temperature : float
        Temperature for text generation. Default: ``1.0``.
    no_repeat_ngram_size : int
        N-gram blocking size for text generation. Default: ``5``.
    """

    BOW_PREFIX = " "
    AUDIO_TOKEN_STRIDE = 640
    AUDIO_TOKEN_INDEX = 151675   # <|audio_pad|>
    AUDIO_START_TOKEN_ID = 151669  # <|audio_start|>
    AUDIO_END_TOKEN_ID = 151670    # <|audio_end|>
    SYSTEM_PROMPT = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
        "perceiving auditory and visual inputs, as well as generating text and speech."
    )

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.bow_prefix = self.BOW_PREFIX
        text_history_cls = class_load(self.text_history_config.type)
        self.text_history_method = text_history_cls(self.text_history_config, self.bow_prefix)
        self.audio_subsampling_factor = self.AUDIO_TOKEN_STRIDE
        self.repetition_penalty = getattr(self.config, "repetition_penalty", 1.05)
        self.temperature = getattr(self.config, "temperature", 1.0)
        self.no_repeat_ngram_size = getattr(self.config, "no_repeat_ngram_size", 5)

    @classmethod
    def load_model(cls, config: SimpleNamespace) -> None:
        """Load the Qwen3-Omni model and processor."""
        model_name = getattr(
            config,
            "hf_model_name",
            getattr(config, "model_path", "Qwen/Qwen3-Omni-30B-A3B-Instruct"),
        )
        attn_impl = getattr(config, "attn_implementation", "eager")

        cls.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=attn_impl,
            enable_audio_output=False,
        )
        cls.processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)
        cls.model.eval()

    def build_prompt(self) -> str:
        """Build the translation instruction used alongside the audio input."""
        return (
            TEMPLATED_SPEECH_PROMPT
            .replace("{src_lang}", LANG_MAPPER.get(self.src_lang, self.src_lang))
            .replace("{tgt_lang}", LANG_MAPPER.get(self.tgt_lang, self.tgt_lang))
        )

    def build_processor_inputs(self, waveform: np.ndarray) -> dict:
        """Build multimodal processor inputs from the rolling audio history."""
        prompt_text = self.build_prompt()
        prefix = self.build_raw_text_prefix()

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

        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)

        inputs = self.processor(
            text=f"{prompt}{prefix}",
            audio=audios,
            images=images,
            videos=videos,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        return inputs.to(self.device).to(self.model.dtype)

    def _find_audio_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token positions corresponding to the encoded audio span."""
        audio_positions = (input_ids[0] == self.AUDIO_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        if audio_positions.numel() > 0:
            return audio_positions

        start_pos = (input_ids[0] == self.AUDIO_START_TOKEN_ID).nonzero(as_tuple=True)[0][0]
        end_pos = (input_ids[0] == self.AUDIO_END_TOKEN_ID).nonzero(as_tuple=True)[0]
        end_pos = end_pos[end_pos > start_pos][0]
        return torch.arange(start_pos + 1, end_pos, device=input_ids.device)

    def _generate(self, inputs: dict) -> Tuple[List[str], torch.Tensor]:
        """
        Run greedy generation and build the proxy cross-attention matrix.

        Qwen3-Omni returns the thinker self-attention scores for each decode
        step and layer. H is the dimension of the attention heads.

        output.attentions[0][layer] -> (1, H, input_len, input_len)  # prefill
        output.attentions[i][layer] -> (1, H, 1, input_len+i)        # new token i

        Returns
        -------
        List[str]
            A list of the newly generated tokens (n_new).
        torch.Tensor
            Proxy cross-attention scores extracted from the self-attention scores
            (prefix + n_new, audio_len).
        """
        input_ids = inputs["input_ids"]  # (1, input_len)
        input_len = input_ids.shape[1]

        # Locate audio positions.
        audio_positions = self._find_audio_positions(input_ids)
        audio_len = audio_positions.shape[0]

        # Generate.
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
            #thinker_eos_token_id=[151643, 151645],
            temperature=self.temperature,
        )
        if isinstance(output, tuple):
            output = output[0]

        # Decode newly generated tokens only.
        new_ids = output.sequences[:, input_len:]
        new_tokens = [
            self.processor.tokenizer.decode([token_id], skip_special_tokens=True)
            for token_id in new_ids[0]
        ]

        # Build proxy cross-attention for the hypothesis (prefix + new_tokens).
        prefill_attn = self.mean_attn_over_heads_and_selected_layers(output.attentions[0])
        prefix_len = len(self.text_history) if self.text_history else 0
        empty_attn = torch.zeros(0, audio_len, device=self.device)

        # Prefix rows come from the prefill pass.
        prefix_rows = prefill_attn[input_len - prefix_len:, :][:, audio_positions] \
            if prefix_len > 0 else empty_attn
        # The prefill pass predicts the first generated token, so its last prompt row
        # is used as the first generated token's proxy audio-attention.
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
        """Convert decoded tokens to the emitted text string."""
        return "".join(tokens)
