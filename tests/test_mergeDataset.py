import tempfile
from unittest import TestCase

import os

from ddt import ddt, data, unpack

from datasetmovies.constants import encoding
from datasetmovies.MergeDataset import MergeDataset


@ddt
class TestMergeDataset(TestCase):

    @data(("data/rt-polarity.pos.txt", "data/rt-polarity.neg.txt"))
    @unpack
    def test_shouldmerge(self, posfile, negfile):
        # Arrange
        posfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), posfile)
        negfile = os.path.join(os.path.dirname(os.path.realpath(__file__)), negfile)

        sut = MergeDataset()


        # Act
        with open(posfile, encoding=encoding) as poshandle:
            poslines = len( poshandle.readlines())
            poshandle.seek(0)

            with open(negfile, encoding=encoding) as neghandle:
                neglines = len( neghandle.readlines())
                neghandle.seek(0)

                with tempfile.TemporaryFile(mode="w+") as outhandle:
                    sut(poshandle, neghandle, outhandle)
                    outhandle.seek(0)
                    outhandle.seek(0)
                    outlines = len(outhandle.readlines())



        # Assert
        self.assertEqual(poslines+neglines, outlines, "The number of lines in the output must match the sum of the input files")

