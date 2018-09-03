from unittest import TestCase

import numpy
from ddt import ddt

from utils.NlpUtils import UNKNOWN_WORD, EOS
from datasetmovies.MovieDatasetIteratorProcessor import MovieDatasetIteratorProcessor


@ddt
class TestMoviesDatasetIteratorProcesor(TestCase):

    def test__getline(self):
        data = [["splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .", 1]
            , ["splash even greater than arnold schwarzenegger ", 0]]

        sut = MovieDatasetIteratorProcessor(data, vocab={"splash": 1, "jean": 2, UNKNOWN_WORD: 0, EOS: 3})

        # Act
        actual = sut[1]

        # Assert
        expected = (numpy.asarray([1, 0, 0, 0, 0, 0, 3], dtype=int), 0)
        self.assertEqual(len(actual), len(expected))
        self.assertSequenceEqual(actual[0].tolist(), expected[0].tolist())
        self.assertEqual(actual[1], expected[1])

    def test__getline_without_vocab(self):
        data = [["splash even", 1]
            , ["splash even greater", 0]]

        sut = MovieDatasetIteratorProcessor(data)

        # Act
        actual = sut[1]

        # Assert
        expected_token_len = 4
        self.assertEqual(len(actual[0]), expected_token_len)

    def test__len(self):
        # Arrange
        data = [["splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .", 1]
            , ["splash even greater than arnold schwarzenegger ", 0]]
        expected_total = len(data)

        sut = MovieDatasetIteratorProcessor(data, vocab={"splash": 1, "jean": 2})

        # Act
        actual = len(sut)

        # Assert
        self.assertEqual(actual, expected_total)
