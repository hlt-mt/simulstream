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
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from qwen_omni_utils import process_mm_info
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.base_doa import DecoderOnlyAttention, get_language_name


class Qwen3OmniDOA(DecoderOnlyAttention):
    """
    Decoder-Only Attention agent for Qwen3-Omni.

    Args:
       config (SimpleNamespace): Configuration object. The following additional attributes are
           expected:
           - **repetition_penalty (float)**: Repetition penalty for text generation.
           Default: ``1.05``.
           - **temperature (float)**: Temperature parameter. Default: ``1.0``.
           - **no_repeat_ngram_size (int)**: Ngram size for text generation. Default: ``5``.
    """

    AUDIO_TOKEN_STRIDE = 640
    AUDIO_TOKEN_INDEX = 151675  # <|audio_pad|>
    AUDIO_START_TOKEN_ID = 151669  # <|audio_start|>
    AUDIO_END_TOKEN_ID = 151670  # <|audio_end|>
    SYSTEM_PROMPT = (
        "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
        "perceiving auditory and visual inputs, as well as generating text and speech."
    )

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
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

        cls.processor = Qwen3OmniMoeProcessor.from_pretrained(model_name)
        cls.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            attn_implementation=attn_impl,
            enable_audio_output=False,
        )
        cls.model.eval()

    def build_prompt(self) -> str:
        """Build the translation instruction used alongside the audio input."""
        return (
            self.prompt
            .replace("{src_lang}", get_language_name(self.src_lang))
            .replace("{tgt_lang}", get_language_name(self.tgt_lang))
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

    def tokens_to_string(self, tokens: List[str]) -> str:
        """Convert decoded tokens to the emitted text string."""
        return "".join(tokens)

    def _find_audio_positions(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return token positions corresponding to the encoded audio span."""
        audio_positions = (input_ids[0] == self.AUDIO_TOKEN_INDEX).nonzero(as_tuple=True)[0]
        if audio_positions.numel() > 0:
            return audio_positions

        start_pos = (input_ids[0] == self.AUDIO_START_TOKEN_ID).nonzero(as_tuple=True)[0][0]
        end_pos = (input_ids[0] == self.AUDIO_END_TOKEN_ID).nonzero(as_tuple=True)[0]
        end_pos = end_pos[end_pos > start_pos][0]
        return torch.arange(start_pos + 1, end_pos, device=input_ids.device)

    def _do_generate(self, inputs: Dict[str, Any]) -> Tuple[List[str], List[torch.Tensor]]:
        input_len = inputs["input_ids"].shape[1]
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
        )
        if isinstance(output, tuple):
            output = output[0]
        new_tokens = self.processor.tokenizer.convert_ids_to_tokens(
            output.sequences[0, input_len:], skip_special_tokens=True)
        return new_tokens, output.attentions
