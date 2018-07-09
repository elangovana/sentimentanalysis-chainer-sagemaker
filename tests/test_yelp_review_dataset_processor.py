from unittest import TestCase
import ddt
import os
from ddt import ddt, data, unpack
from custom_chainer.yelp_review_dataset_processor import YelpReviewDatasetProcessor


@ddt
class TestYelpReviewDatasetProcessor(TestCase):

    @data(("data/yelp_review_short.csv", 3))
    @unpack
    def test_read_data(self, file, expected_no_lines):
         #Arrange
         full_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), file)
         sut = YelpReviewDatasetProcessor()

         # Act
         actual = sut.read_data(full_file_path)

         #Assert
         self.assertEqual(expected_no_lines, len(actual))
