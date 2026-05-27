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

import os
import sys
import tempfile
import types
import unittest

try:
    import websockets  # noqa: F401
except ModuleNotFoundError:
    websockets = types.ModuleType("websockets")
    websockets.ClientConnection = object
    sys.modules["websockets"] = websockets

from simulstream.client.wav_reader_client import load_wav_file_list


class LoadWavFileListTestCase(unittest.TestCase):

    def _write_wav_list(self, tmpdir, wav_list_entry):
        wav_path = os.path.join(tmpdir, "audio.wav")
        list_path = os.path.join(tmpdir, "wav_files.txt")
        with open(wav_path, "w"):
            pass
        with open(list_path, "w") as f:
            f.write(f"{wav_list_entry}\n")
        return wav_path, list_path

    def test_loads_relative_path_from_list_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path, list_path = self._write_wav_list(tmpdir, "audio.wav")

            wav_files = load_wav_file_list(list_path)

            self.assertEqual(wav_files, [wav_path])

    def test_loads_absolute_path_from_list_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path, list_path = self._write_wav_list(
                tmpdir, os.path.join(tmpdir, "audio.wav"))

            wav_files = load_wav_file_list(list_path)

            self.assertEqual(wav_files, [wav_path])


if __name__ == "__main__":
    unittest.main()
