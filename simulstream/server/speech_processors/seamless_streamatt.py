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
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoProcessor, SeamlessM4TModel, SeamlessM4Tv2Model

from simulstream.server.speech_processors import SAMPLE_RATE
from simulstream.server.speech_processors.base_streamatt import BaseStreamAtt


SHIFT_SIZE = 10
SEAMLESS_AUDIO_SUBSAMPLING_FACTOR = 8
# `FRAME_LENGTH` is the length of the window used for computing the mel-filterbank features that,
# with a sample rate of 16kHz, corresponds to 25ms while `HOP_LENGTH` is how much the feature
# computation is shifted at each step, corresponding to 10ms. Values from `transformers.models.
# seamless_m4t.feature_extraction_seamless_m4t.SeamlessM4TFeatureExtractor._extract_fbank_features`
FRAME_LENGTH = 400
HOP_LENGTH = 160
# `OVERLAP_WINDOW` is length of the features that overlap at each step, as described in
# `transformers.audio_utils.spectrogram`, and corresponding to 15ms.
OVERLAP_WINDOW = FRAME_LENGTH - HOP_LENGTH


class SeamlessStreamAtt(BaseStreamAtt):
    """
    Perform StreamAtt policy with the chosen textual history selection using a SeamlessM4T
    speech-to-text model.

    Args:
       config (SimpleNamespace): Configuration object.
           The following additional attributes are expected:
           - **max_new_tokens (int)**: The maximum numbers of tokens to generate
           - **num_beams (int)**: The number of beams of the beam search
           - **no_repeat_ngram_size (int)**: The maximum number of ngram repeats that
             cannot be emitted
           - **audio_history_max_duration (int)**: Maximum length of the audio history to store,
             in seconds. Defaults to 1 hour.

    """

    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.audio_subsampling_factor = SEAMLESS_AUDIO_SUBSAMPLING_FACTOR
        self.max_new_tokens = getattr(self.config, "max_new_tokens", 128)
        self.num_beams = getattr(self.config, "num_beams", 5)
        self.audio_history_max_duration = getattr(self.config, "audio_history_max_duration", 360)
        self.no_repeat_ngram_size = getattr(self.config, "no_repeat_ngram_size", 5)
        self.waveform_accumulator = None

    @property
    def audio_max_len(self) -> float:
        """
        Returns the maximum allowed length for the audio history converted in the space of
        audio features. The number of encoded features is obtained by first converting the
        self.audio_history_max_duration, originally in seconds, into milliseconds, and then
        by the dimension of the shift (SHIFT_SIZE). Since the SeamlessM4t encoder subsamples
        the input sequence, the resulting *audio_max_len* is obtained by further dividing
        the original sequence length by *self.audio_subsampling_factor*.
        """
        return self.audio_history_max_duration * 1000 // SHIFT_SIZE // \
            self.audio_subsampling_factor

    @classmethod
    def load_model(cls, config: SimpleNamespace):
        if not hasattr(cls, "model") or cls.model is None:
            cls.processor = AutoProcessor.from_pretrained(config.hf_model_name)
            seamless_version = getattr(config, "seamless_version", 1)
            if seamless_version == 2:
                cls.model = SeamlessM4Tv2Model.from_pretrained(config.hf_model_name)
            else:
                cls.model = SeamlessM4TModel.from_pretrained(config.hf_model_name)
            cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls.model.to(cls.device)

    @staticmethod
    def mean_variance_normalization(features: np.ndarray) -> torch.Tensor:
        """
        Normalization function taken from `transformers.models.seamless_m4t.
        feature_extraction_seamless_m4t.SeamlessM4TFeatureExtractor`.
        """
        # torch defaults to ddof=1, and numpy defaults to ddof=0
        mu = np.expand_dims(features.mean(axis=0), 0)
        sigma = np.sqrt(np.expand_dims(features.var(axis=0, ddof=1), 0) + 1e-7)
        normalized = (features - mu) / sigma
        return torch.tensor(np.array(normalized))

    def _preprocess(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Extract normalized input features for the SeamlessM4T model from the new
        audio chunk (`waveform`) and the overlapping tail contained in
        `self.waveform_accumulator`,  and then concatenating them with previously
        extracted features stored in `self.audio_history`.

        Steps:
        1. Convert the new waveform and the overlapping tail into mel-filterbank features.
        2. Concatenate the new features with any stored `self.audio_history`.
        3. Apply mean-variance normalization.
        4. Cache the overlapping tail of the waveform for the next step.

        Returns:
            torch.Tensor: Normalized feature tensor on `self.device`.
        """
        # Combine with overlapping part from previous step, if any
        if self.waveform_accumulator is not None:
            waveform = np.concatenate((self.waveform_accumulator, waveform))

        if len(waveform) >= FRAME_LENGTH:
            # Extract new mel-filterbank features (unnormalized)
            new_features = self.processor(
                audios=waveform,
                return_tensors="np",
                do_normalize_per_mel_bins=False,
                sampling_rate=SAMPLE_RATE,
            )["input_features"].squeeze(0)  # shape: (T_new, 160)

            # Concatenate with previous features, if available
            if self.audio_history is not None:
                self.audio_history = np.concatenate((self.audio_history, new_features), axis=0)
            else:
                self.audio_history = new_features

        # Normalize all features
        normalized_features = self.mean_variance_normalization(self.audio_history)

        # Store the last part of the waveform for the next preprocessing step
        self.waveform_accumulator = waveform[-OVERLAP_WINDOW:]

        return normalized_features.unsqueeze(0).to(self.device)

    def get_prefix(self):
        """
        Creates a prefix for the generation phase with the previous outputs stored in
        the text_history. The prefix is formatted following the SeamlessM4T template
        (tgt language token id + text history ids).
        """
        prefix_ids = torch.tensor(
            [[self.model.generation_config.text_decoder_lang_to_code_id.get(self.tgt_lang)]]
        ).to(self.device)
        if self.text_history:
            text_history_ids = torch.tensor(
                self.processor.tokenizer.convert_tokens_to_ids(self.text_history)).unsqueeze(
                dim=0).to(self.device)
            prefix_ids = torch.cat((prefix_ids, text_history_ids), dim=1)
        return prefix_ids.long()

    def _generate(
            self, input_features: torch.Tensor, normalize_attn: bool = True
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Generates a new hypothesis returning also the cross attention scores.
        The hypothesis is forced to start with the `prefix_ids` prefix tokens,
        which are then removed from the returned output tokens.
        The attention scores are returned with dimensions (sequence_len, n_audio_features)
        where `sequence_len` is the length of the prefix (excluding the language ID)
        and of the new hypothesis.
        """
        prefix_ids = self.get_prefix()
        gen_out = self.model.generate(
            input_features=input_features,
            decoder_input_ids=prefix_ids,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            return_dict_in_generate=True,
            output_attentions=True,
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            generate_speech=False)
        out_ids = list(gen_out.sequences[0])

        # Exclude BOS, prefix, and EOS from the generated sequence
        new_hypo_ids = out_ids[prefix_ids.shape[1] + 1:-1]

        cross_attn = self.get_cross_attention(
            gen_out, len(new_hypo_ids), normalize_attn=normalize_attn)

        assert cross_attn.shape[0] == (prefix_ids.shape[1] - 1) + len(new_hypo_ids), \
            f"Cross attention scores along tokens dimension ({cross_attn.shape[0]}) " \
            f"mismatch with the length of the hypothesis " \
            f"({(prefix_ids.shape[1] - 1) + len(new_hypo_ids)})."

        new_hypo = self.processor.tokenizer.convert_ids_to_tokens(
            new_hypo_ids, skip_special_tokens=True)
        return new_hypo, cross_attn

    def _extract_new_hypo_attention_scores(self, new_hypo_len: int, gen_out):
        """
        Extract attention scores `cross_attentions` from `gen_out`, which are stored for each
        generation step and Layer `layer`, based on the beam index selected at each step of the
        beam search and stored in `beam_indices` (if num_beams > 1) for the new hypotheses.
        """
        cross_attns = []
        if self.num_beams > 1:
            # Beam search: for each token of the new hypothesis, we select the corresponding cross
            # attention from the cross attentions stored at each step of the beam search using the
            # index contained in the tensor of indices beam_indices (num_beams * sequence length)
            for tok_idx in range(new_hypo_len):
                # Select the cross attention matrix using the beam_indices
                beam_indices = gen_out.beam_indices[:, tok_idx]
                # cross_attentions contains the forced decoder prefix attention at index 0.
                # The cross attention for the first token (i.e. tok_idx = 0) is at index 1, and so on.
                # -1 selects the current token query row for that step
                cross_attn = gen_out.cross_attentions[
                    tok_idx + 1][self.cross_attn_layer][:, :, -1, :]
                cross_attn = cross_attn.index_select(dim=0, index=beam_indices)
                cross_attns.append(cross_attn)
        else:
            # Greedy search
            for tok_idx in range(new_hypo_len):
                # See the beam-search branch above for why tok_idx + 1 and -1 are used.
                cross_attn = gen_out.cross_attentions[
                    tok_idx + 1][self.cross_attn_layer][:, :, -1, :]
                cross_attns.append(cross_attn)

        # Cross attention scores with shape [num_heads, sequence_len, n_audio_features]
        cross_attns = torch.stack(cross_attns).squeeze(1)
        return cross_attns

    def get_cross_attention(
            self, gen_out, new_hypo_len: int, normalize_attn: bool) -> torch.Tensor:
        """
        Given the attention scores for the generated output (including both prefix and new
        hypothesis), this function returns the cross attention scores from Layer *layer* by
        averaging the scores along the attention heads dimension.
        """
        # The prefix is stored in the first element of the cross_attentions, equal for each beam
        # and, therefore, the first is selected. Also, BOS and language ID are excluded from the
        # sequence_len (first two elements).
        cross_attns = (
            gen_out.cross_attentions[0][self.cross_attn_layer][0, :, 2:, :].transpose(0, 1))

        if new_hypo_len > 0:
            new_cross_attns = self._extract_new_hypo_attention_scores(new_hypo_len, gen_out)
            cross_attns = torch.cat([cross_attns, new_cross_attns], dim=0)

        # Average on the attention heads dimension
        cross_attns = cross_attns.mean(dim=1)

        # Normalize attention scores
        if normalize_attn:
            cross_attns = self.normalize_attn(cross_attns)
        return cross_attns

    def tokens_to_string(self, tokens: List[str]) -> str:
        # avoid that the initial space, if it is there, get removed in the detokenization
        if self.text_history is not None and len(self.text_history) > 0:
            tokens = [''] + tokens
        return self.processor.tokenizer.convert_tokens_to_string(tokens)

    def set_target_language(self, language: str) -> None:
        self.tgt_lang = language

    def set_source_language(self, language: str) -> None:
        pass

    def clear(self) -> None:
        super().clear()
        self.waveform_accumulator = None
