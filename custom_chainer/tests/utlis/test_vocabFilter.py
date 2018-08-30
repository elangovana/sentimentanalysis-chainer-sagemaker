from unittest import TestCase
from ddt import ddt, data, unpack
from utils.VocabFilter import VocabFilter


@ddt
class TestVocabFilter(TestCase):

    @data(("black", 2, 10, True, False)
        , ("jack", 2, 1, False, True)
        , ("memo", 2, 3, False, True)
        , ("unknown", 2, 1, False, False)
        , ("unknown", 2, 1, True, True))
    @unpack
    def test__call(self, word, min_freq, max_vocab_size, filter_unknown_words, expected):
        tokens_count_dict = {"black": 10, "jack": 5, "memo": 1, "monk": 10, "label": 15}
        sut = VocabFilter(tokens_count_dict=tokens_count_dict, min_frequency=min_freq, max_vocab_size=max_vocab_size,
                          filter_unknown_words=filter_unknown_words)

        # Act
        actual = sut(word)

        # Assert
        self.assertEqual(actual, expected)
