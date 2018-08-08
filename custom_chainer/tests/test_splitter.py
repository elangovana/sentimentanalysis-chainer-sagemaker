import tempfile
from unittest import TestCase
import os
from ddt import ddt, data, unpack

from dataprep.Splitter import Splitter


@ddt
class TestSplitter(TestCase):

    @data(("data/parts/yelp_review_short.csv", 3, "data/parts"))
    @unpack
    def test_split(self, inputfile, no_of_parts, expected_parts_dir):
        # Arrange
        full_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), inputfile)
        expected_parts_full_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), expected_parts_dir)

        temp_out_dir ="/Users/aeg/Documents/sentimentanalysis-chainer-sagemaker/custom_chainer/tests/data/parts/test"# tempfile.mkdtemp()

        sut = Splitter(inputfile, temp_out_dir)

        # Act
        sut.split(temp_out_dir, no_of_parts=no_of_parts)

        #Assert
        for f in os.listdir(temp_out_dir):
            actual_f = os.path.join(temp_out_dir, f)
            expected_f = os.path.join(expected_parts_full_path, f)

            with open(actual_f, "r") as f:
                actual = f.read()

            with open(expected_f, "r") as f:
                expected = f.read()

            self.assertTrue(actual == expected, "\n{}\n\n{}".format(actual,expected) )

