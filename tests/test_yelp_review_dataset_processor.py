from unittest import TestCase
import os
from ddt import ddt, data, unpack
from yelpdataset.YelpReviewDatasetProcessor import YelpReviewDatasetProcessor


@ddt
class TestYelpReviewDatasetProcessor(TestCase):


    def should_get_length(self):
        # Arrange
        iterator = [["vkVSCC7xljjrAI4UGfnKEQ", "bv2nCi5Qv5vroFiqKGopiw", "AEx2SYEUJmTxVVB18LlCwA", "5", "2016-05-28",
                     "Super simple place but amazing nonetheless. It's been around since the 30's and they still serve the same thing they started with: a bologna and salami sandwich with mustard.",
                     "0", "0", "0"]
            , ["n6QzIUObkYshz4dz2QRJTw", "bv2nCi5Qv5vroFiqKGopiw", "VR6GpWIda3SfvPC-lg9H3w", "5", "2016-05-28",
               "Small unassuming place that changes their menu every so often. Cool decor and vibe inside their 30 seat restaurant. Call for a reservation",
               "0", "0", "0"]
            , ["MV3CcKScW05u5LVfF6ok0g", "bv2nCi5Qv5vroFiqKGopiw", "CKC0-MOWMqoeWf6s-szl8g", "5", "2016-05-28",
               "Lester's is located in a beautiful neighborhood and has been there since 1951. They are known for smoked meat which most deli's have but their brisket sandwich is what I come to montreal for",
               "0", "0", "0"]]
        sut = YelpReviewDatasetProcessor(iterator)
        expected_no_lines = len(iterator)

        # Act
        actual = len(sut)

        # Assert
        self.assertEqual(actual, expected_no_lines)
