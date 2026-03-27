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

from simulstream.server.speech_processors.base_streamatt import (
    PunctuationTextHistory, BaseStreamAtt)


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


class TestStripIncompleteWords(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace()
        self._strip_incomplete_words = BaseStreamAtt._strip_incomplete_words

    def test_incomplete_word_is_stripped(self):
        """Last word has no closing token — should be dropped."""
        stripped = self._strip_incomplete_words(self, ["▁U", "ser", "▁Inter", "ac"])
        self.assertEqual(stripped, ["▁U", "ser"])

    def test_single_incomplete_word_returns_empty(self):
        """Only one word and it's incomplete — nothing left to return."""
        stripped = self._strip_incomplete_words(self, ["▁Inter", "ac"])
        self.assertEqual(stripped, [])

    def test_multiple_incomplete_tokens_all_stripped(self):
        """Several continuation tokens after the last BOW — all should be dropped."""
        stripped = self._strip_incomplete_words(self, ["▁U", "ser", "▁Inter", "ac", "ti"])
        self.assertEqual(stripped, ["▁U", "ser"])

    def test_ends_with_period_kept(self):
        """Trailing period counts as strong punctuation — full token list returned."""
        stripped = self._strip_incomplete_words(self, ["▁U", "ser", "▁Inter", "ac", "tion", "."])
        self.assertEqual(stripped, ["▁U", "ser", "▁Inter", "ac", "tion", "."])

    def test_ends_with_multiple_periods(self):
        """Trailing period counts as strong punctuation — full token list returned."""
        stripped = self._strip_incomplete_words(
            self, ["▁U", "ser", "▁Inter", "ac", "tion", ".", ".", "."])
        self.assertEqual(stripped, ["▁U", "ser", "▁Inter", "ac", "tion", ".", ".", "."])

    def test_ends_with_non_strong_punctuation(self):
        """Non strong punctuation marks should be treated as standard tokens."""
        stripped = self._strip_incomplete_words(self, ["▁Hello", "-"])
        self.assertEqual(stripped, [])

    def test_ends_with_question_mark(self):
        """Question marks should be treated as strong punctuation."""
        stripped = self._strip_incomplete_words(self, ["▁Is", "▁this", "▁work", "ing", "?"])
        self.assertEqual(stripped, ["▁Is", "▁this", "▁work", "ing", "?"])

    def test_trailing_empty_token_stripped_before_check(self):
        """Empty trailing tokens should be dropped; remaining punctuation keeps the list intact."""
        stripped = self._strip_incomplete_words(self, ["▁output", ".", ""])
        self.assertEqual(stripped, ["▁output", "."])

    def test_multiple_trailing_empty_tokens(self):
        """Multiple trailing empty tokens should be dropped."""
        stripped = self._strip_incomplete_words(self, ["▁Hello", ".", "", ""])
        self.assertEqual(stripped, ["▁Hello", "."])

    def test_only_empty_tokens_returns_empty(self):
        """Only empty tokens should be dropped."""
        stripped = self._strip_incomplete_words(self, ["", "", ""])
        self.assertEqual(stripped, [])

    def test_empty_input(self):
        """Empty input should return an empty list."""
        stripped = self._strip_incomplete_words(self, [])
        self.assertEqual(stripped, [])

    def test_single_bow_token_incomplete(self):
        """A lone BOW token with no following token is itself incomplete."""
        stripped = self._strip_incomplete_words(self, ["▁Hello"])
        self.assertEqual(stripped, [])

    def test_no_bow_prefix_at_all(self):
        """No BOW token anywhere — loop never breaks, returns empty list."""
        stripped = self._strip_incomplete_words(self, ["ac", "tion"])
        self.assertEqual(stripped, [])


if __name__ == "__main__":
    unittest.main()
