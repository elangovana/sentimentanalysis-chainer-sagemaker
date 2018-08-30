from unittest import TestCase
from ddt import ddt, data, unpack
from utils.VocabFilter import VocabFilter


@ddt
class TestVocabFilter(TestCase):

    @data(("black", 2, 10,  False)
        , ("jack", 2, 1,  True)
        , ("memo", 2, 3, False)
        , ("unknown", 2, 1, False)
          )
    @unpack
    def test__call(self, word, min_freq, max_vocab_size,  expected):
        tokens_count_dict = {"black": 10, "jack": 5, "memo": 1, "monk": 10, "label": 15}
        sut = VocabFilter(tokens_count_dict=tokens_count_dict, min_frequency=min_freq, max_vocab_size=max_vocab_size,
                          priority_words={"memo", "unknown", "eof"})

        # Act
        actual = sut(word)

        # Assert
        self.assertEqual(actual, expected)
