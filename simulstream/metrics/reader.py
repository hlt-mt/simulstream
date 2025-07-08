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

import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import List, Any, Dict

from simulstream.metrics.detokenizers import get_detokenizer


class LogReader:
    """
    Helper class to read metric logs.
    """
    def __init__(self, config: SimpleNamespace, filepath: str):
        self.filepath = filepath
        self.detokenizer = get_detokenizer(config)
        self.outputs_by_audio = self._get_outputs()

    def _get_outputs(self) -> Dict[str, List[Dict[str, Any]]]:
        outputs_by_audio = OrderedDict()
        audio_id_map = {}
        for line in self._read_all():
            if 'metadata' in line:
                audio_id_map[line['id']] = Path(line['metadata']['wav_name']).stem
            else:
                assert line['id'] in audio_id_map, \
                    f'{line["id"]} not associated with audio file'
                audio_name = audio_id_map[line['id']]
                if audio_name not in outputs_by_audio:
                    outputs_by_audio[audio_name] = []
                outputs_by_audio[audio_name].append(line)
        return outputs_by_audio

    def _read_all(self) -> List[Any]:
        data = []
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # skip empty lines
                    data.append(json.loads(line))
        return data

    def final_outputs(self) -> Dict[str, str]:
        """
        Returns the final tokenized outputs for each audio. This means that overridden tokens
        in retranslation are not included in the output, which is the final string obtained at
        the end of the audio file. The output string is tokenized, so it should be detokenized
        before usage.
        """
        outputs: OrderedDict[str, str] = OrderedDict()
        for audio, lines in self.outputs_by_audio.items():
            tokens = []
            for line in lines:
                if len(line['deleted_tokens']) > 0:
                    assert line['deleted_tokens'] == tokens[-len(line['deleted_tokens']):]
                    tokens = tokens[:-len(line['deleted_tokens'])]
                tokens.extend(line['generated_tokens'])
            outputs[audio] = self.detokenizer(' '.join(tokens))
        return outputs


class ReferenceReader:
    def __init__(self, references: List[str]):
        self.references = self._read_all(references)

    @staticmethod
    def _read_all(references: List[str]) -> Dict[str, List[str]]:
        reference_by_file = OrderedDict()
        for reference in references:
            with open(reference, 'r', encoding='utf-8') as f:
                reference_by_file[Path(reference).stem] = f.readlines()
        return reference_by_file

    def get_all(self) -> Dict[str, List[str]]:
        return self.references
