from unittest import TestCase
import os

import numpy
from ddt import ddt, data, unpack

from NlpUtils import UNKNOWN_WORD, EOS
from dataprep.YelpChainerDataset import YelpChainerDataset
from moviesdataset.MovieDatasetIteratorProcessor import MovieDatasetIteratorProcessor


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

    def test_getcount(self):
        # Arrange
        data = [["splash even greater than arnold schwarzenegger , jean-claud van damme or steven segal .", 1]
            , ["splash even greater than arnold schwarzenegger ", 0]]
        expected_total = len(data)

        sut = MovieDatasetIteratorProcessor(data, vocab={"splash": 1, "jean": 2})

        # Act
        actual = len(sut)

        # Assert
        self.assertEqual(actual, expected_total)
