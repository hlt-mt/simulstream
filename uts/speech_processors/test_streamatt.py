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

import unittest
from types import SimpleNamespace

from simulstream.server.speech_processors.base_streamatt import PunctuationTextHistory


class TestPunctuationTextHistory(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace()
        self.punctuation_text_history = PunctuationTextHistory(self.config)

    def test_punctuation_last(self):
        """ Test PunctuationTextHistory method when the history ends with strong punctuation. """
        # Test word level
        en_history = ["Hi", "!", "I", "am", "Sara", "."]
        selected_history = self.punctuation_text_history.select_text_history(en_history)
        self.assertEqual(selected_history, ["I", "am", "Sara", "."])

        # Test character level
        zh_history = ['担', '任', '开', '发', '主', '管', '。']
        selected_history = self.punctuation_text_history.select_text_history(zh_history)
        self.assertEqual(selected_history, ['担', '任', '开', '发', '主', '管', '。'])

    def test_punctuation_in_between(self):
        """ Test PunctuationTextHistory method when punctuation separates two sentences. """
        # Test word level
        en_history = ["Hi", "!", "I", "am", "Sara"]
        selected_history = self.punctuation_text_history.select_text_history(en_history)
        self.assertEqual(selected_history, ["I", "am", "Sara"])

        # Test character level
        zh_history = ['开', '发', '主', '管', '。', '担', '任']
        selected_history = self.punctuation_text_history.select_text_history(zh_history)
        self.assertEqual(selected_history, ['担', '任'])

    def test_no_strong_punctuation(self):
        """ Test PunctuationTextHistory method when no strong punctuation is present. """
        # Test word level
        en_history = ["Hi", ",", "I", "am", "Sara"]
        selected_history = self.punctuation_text_history.select_text_history(en_history)
        self.assertEqual(selected_history, ["Hi", ",", "I", "am", "Sara"])

        # Test character level
        zh_history = ['回', '到', '纽', '约', '后', '，', '我']
        selected_history = self.punctuation_text_history.select_text_history(zh_history)
        self.assertEqual(selected_history, ['回', '到', '纽', '约', '后', '，', '我'])


if __name__ == "__main__":
    unittest.main()
