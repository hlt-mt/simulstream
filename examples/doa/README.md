# Decoder-Only Attention (DOA) Policy

The [**Decoder-Only Attention (DOA)** policy](https://arxiv.org/abs/2605.31432) extends the
[encoder-decoder **StreamAtt** policy](https://aclanthology.org/2024.acl-long.202/) to SpeechLLMs
that have no cross-attention mechanism. Instead of relying on encoder-decoder cross-attention, DOA
builds a *proxy* cross-attention matrix by extracting the self-attention weights between the audio
tokens and the text tokens from the decoder layers. The resulting matrix is then used by the
[AlignAtt policy](https://www.isca-archive.org/interspeech_2023/papi23_interspeech.html) to decide
which generated tokens can be safely emitted at each step.

## Supported models

| Class | HuggingFace model |
|---|---|
| `simulstream.server.speech_processors.phi4multimodal_doa.Phi4MultimodalDOA` | `microsoft/Phi-4-multimodal-instruct` |
| `simulstream.server.speech_processors.qwenomni_doa.Qwen3OmniDOA` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |

DOA is supported in the `simulstream_inference` backend.

## Configuration

A DOA config file is a YAML file passed via `--speech-processor-config`. Below is a full annotated 
example, elements between `{}` brackets should be replaced as mentioned below:

```yaml
type: "{MODEL_CLASS}"
text_history:
  type: "simulstream.server.speech_processors.base_streamatt.{TEXT_HISTORY}"
audio_history_max_duration: 180
text_history_max_len: 128
speech_chunk_size: 1              # seconds of audio fed to the model at each step
max_new_tokens: 32                # max tokens generated per chunk
hf_model_name: "{MODEL_NAME}"
detokenizer_type: "hf"            
word_level_postprocess: {WORD_POSTPROCESS}      
bow_prefix: "{BOW_MARKER}"                   
prompt: "{PROMPT}"

# --- DOA parameters ---
attn_layer: {ATTN_LAYER}
attn_head: {ATTN_HEAD}
average_attn_over_layers:  {ATTN_AVG}
cutoff_frame_num: {CUTOFF_FRAME}
```

Parameters to be replaced:
- `{MODEL_CLASS}` from the [supported models Class](#supported-models) 
(e.g., `simulstream.server.speech_processors.phi4multimodal_doa.Phi4MultimodalDOA`)
- `{TEXT_HISTORY}` among:
  - `FixedWordsTextHistory`: Retains the last *N* complete words. Recommended for space-separated 
languages (English, Italian, …). In this case, the number of `history_words` should be added to 
the config, for instance:
  ```yaml
  text_history:
    type: "simulstream.server.speech_processors.base_streamatt.FixedWordsTextHistory"
    history_words: 10
  ```
  - `FixedCharsTextHistory`: Retains the last *N* characters. Recommended for character-level 
languages (Chinese, Japanese) where `FixedWordsTextHistory` is ineffective because spaces are 
sparse. n this case, the number of `history_chars` should be added to the config, for instance:
  ```yaml
  text_history:
    type: "simulstream.server.speech_processors.base_streamatt.FixedWordsTextHistory"
    history_chars: 20
  ```
  - `PunctuationTextHistory`: Retains the text from the last strong punctuation mark (`.`, `!`, 
`?`, `:`, `;`, `。`). Works for both space-separated and character-level languages. 
- `{MODEL_NAME}` from the [supported models HuggingFace model name](#supported-models) (e.g., 
`microsoft/Phi-4-multimodal-instruct`)
- `{WORD_POSTPROCESS}`: When `true`, the output is trimmed to complete words before emission.
Set to `false` for character-level languages (Chinese, Japanese).
- `{BOW_MARKER}`: The beginning-of-word (BOW) marker used by the model's tokenizer (e.g., 
Phi-4-multimodal and Qwen3-Omni use a plain space `" "`).
- `{PROMPT}`: User prompt. The default is `"Translate the audio to {tgt_lang}:"`. It can be 
overridden with the `prompt` key in the yaml. The placeholders `{src_lang}` and `{tgt_lang}` are 
filled in automatically from the language codes passed at inference time.
- `{ATTN_LAYER}`: Decoder layer to extract self-attention from (int, 0-indexed).
- `{ATTN_HEAD}`: Attention head to extract self-attention from (int, 0-indexed). `null` averages 
over all heads. 
- `{ATTN_AVG}`: If `true` (default), average the selected per-layer attention view across all 
layers; `attn_layer` is used only when this is `false`. 
- `{CUTOFF_FRAME}`: Cutoff frame of the AlignAtt policy. Tokens whose attention peak falls in the 
last *N* audio frames are withheld. Higher values add more latency but reduce the risk of cutting 
correct tokens.

## Configurations of DOA's paper

The configurations used to report the final results in Figure 3 are reported below:

### Phi4-Multimodal
```yaml
type: "simulstream.server.speech_processors.phi4multimodal_doa.Phi4MultimodalDOA"
text_history:
  type: "simulstream.server.speech_processors.base_streamatt.PunctuationTextHistory"
audio_history_max_duration: 180
text_history_max_len: 128
speech_chunk_size: 1 
detokenizer_type: "hf"
hf_model_name: "microsoft/Phi-4-multimodal-instruct"
word_level_postprocess: True  
max_new_tokens: 32
bow_prefix: " "
attn_layer: 0
attn_head: null  
average_attn_over_layers: True
cutoff_frame_num: __FRAME__
```

### Qwen3-Omni
```yaml
type: "simulstream.server.speech_processors.qwenomni_doa.Qwen3OmniDOA"
text_history:
  type: "simulstream.server.speech_processors.base_streamatt.PunctuationTextHistory"
audio_history_max_duration: 60
text_history_max_len: 128
speech_chunk_size: 1
detokenizer_type: "hf"
hf_model_name: "Qwen/Qwen3-Omni-30B-A3B-Instruct"
word_level_postprocess: True
max_new_tokens: 32
bow_prefix: " "
prompt: "You are a professional {src_lang}-to-{tgt_lang} translator. Your goal is to accurately \
convey the meaning and nuances of the original {src_lang} speech while adhering to {tgt_lang} \
grammar, vocabulary, and cultural sensitivities. Use precise terminology and a tone appropriate \
for academic or instructional materials. Produce only the {tgt_lang} translation, without any \
additional explanations or commentary. Please translate the provided {src_lang} speech into \
{tgt_lang}:"
attn_layer: 0
attn_head: null
average_attn_over_layers: True
cutoff_frame_num: __FRAME__
```
To run the inference, `__FRAME__` should be replaced with `sed`:
```bash
simulstream_inference --speech-processor-config <(sed "s/__FRAME__/${FRAME}/g" ${CONFIG_YAML}) \
        --wav-list-file ${AUDIOPATH_LIST} \
        --tgt-lang $TGTLANG --src-lang en \
        --metrics-log-file ${OUTLOG}
```
where `${FRAME}` is `5`, `10`, or `15` following the paper, `${CONFIG_YAML}` is the path to the 
aforementioned configuration yaml file, `${AUDIOPATH_LIST}` is the list of test audio files path,
and `${OUTLOG}` is the path to the jsonl output log file.

## Adding a new model

To support a new SpeechLLM, subclass `DecoderOnlyAttention` 
(`simulstream.server.speech_processors.base_doa.DecoderOnlyAttention`) and implement:

- `load_model(config)` — load the model and processor.
- `build_prompt()` — return the text prompt string.
- `build_processor_inputs(waveform)` — build processor inputs from the rolling audio history.
- `_do_generate(inputs)` — run generation and return `(new_tokens, attentions)`.
- `_find_audio_positions(input_ids)` — return the positions of audio tokens in the input sequence.

## Citation
If you use DOA in your work, please cite:

```bibtex
@article{papi-2026-doa,
  title     = {{DOA}: Training-Free Decoder-Only Attention Policy for Long-Form
               Simultaneous Translation with {SpeechLLMs}},
  author    = {Papi, Sara and Bentivogli, Luisa},
  journal   = {arXiv preprint arXiv:2605.31432},
  year      = {2026},
}
```