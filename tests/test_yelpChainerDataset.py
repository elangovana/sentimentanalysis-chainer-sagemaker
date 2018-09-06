from unittest import TestCase
import os
from ddt import ddt, data, unpack

from datasetyelp.YelpChainerDataset import YelpChainerDataset


@ddt
class TestTelpChainerDataset(TestCase):

    @data(("data/yelp_review_short.csv", 2, True, ",", '"', True,
           ["MV3CcKScW05u5LVfF6ok0g","bv2nCi5Qv5vroFiqKGopiw","CKC0-MOWMqoeWf6s-szl8g","5","2016-05-28","Lester's is located in a beautiful neighborhood and has been there since 1951. They are known for smoked meat which most deli's have but their brisket sandwich is what I come to montreal for","0","0","0" ]
           )
          ## Second test
        , ("data/yelp_review_short.csv", 2, True, ",", '"', False,
           ["MV3CcKScW05u5LVfF6ok0g", "bv2nCi5Qv5vroFiqKGopiw", "CKC0-MOWMqoeWf6s-szl8g", "5", "2016-05-28",
            "Lester's is located in a beautiful neighborhood and has been there since 1951. They are known for smoked meat which most deli's have but their brisket sandwich is what I come to montreal for",
            "0", "0", "0"]

           )
          ## 3rd test
        , ("data/yelp_review_short_no_header.csv", 2, False, ",", '"', False,
           ["MV3CcKScW05u5LVfF6ok0g", "bv2nCi5Qv5vroFiqKGopiw", "CKC0-MOWMqoeWf6s-szl8g", "5", "2016-05-28",
            "Lester's is located in a beautiful neighborhood and has been there since 1951. They are known for smoked meat which most deli's have but their brisket sandwich is what I come to montreal for",
            "0", "0", "0"]

           )
          ## fourth test
        , ("data/yelp_review_short_no_header.csv", 2, False, ",", '"', True,
           ["MV3CcKScW05u5LVfF6ok0g", "bv2nCi5Qv5vroFiqKGopiw", "CKC0-MOWMqoeWf6s-szl8g", "5", "2016-05-28",
            "Lester's is located in a beautiful neighborhood and has been there since 1951. They are known for smoked meat which most deli's have but their brisket sandwich is what I come to montreal for",
            "0", "0", "0"]

           )
          )
    @unpack
    def test__getline(self, file, index, has_header, delimiter, quote_character, inmemory, expected_line):
        # Arrange
        full_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), file)
        sut = YelpChainerDataset(full_file_path, has_header=has_header, use_in_memory_shuffle=inmemory,
                                 delimiter=delimiter, quote_character=quote_character)

        # Act
        actual = sut[index]

        # Assert
        self.assertEqual(actual, expected_line)

    @data(("data/yelp_review_short.csv", True, 3)
        , ("data/yelp_review_short_no_header.csv", False, 3)
          )
    @unpack
    def test___getlen(self, file, has_header, expected_total):
        # Arrange
        full_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), file)
        sut = YelpChainerDataset(full_file_path, has_header=has_header)

        # Act
        actual = len(sut)

        # Assert
        self.assertEqual(actual, expected_total)
