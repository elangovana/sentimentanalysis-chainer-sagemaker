from unittest import TestCase
from ddt import ddt
from utils.VocabFilter import VocabFilter


@ddt
class TestVocabFilter(TestCase):

    def test__call(self):
        vocab = {"word1": 1, "word2": 2}
        tokens_count_dict = {"word2": 10, "word3": 5, "word1": 1}

        sut = VocabFilter(tokens_count_dict=tokens_count_dict, min_frequency=2)

        # Act
        actual = sut(vocab=vocab)

        # Assert
        expected = {"word2": 2}
        self.assertEqual(actual, expected)
