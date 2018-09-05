from io import StringIO
from unittest import TestCase

from dataformatter.blazingTextFormatter import BlazingTextFormatter


class TestBlazingTextFormatter(TestCase):

    def test_format(self):
        # Arrange
        output = StringIO()
        sut = BlazingTextFormatter()
        # Arrange
        iterator = [["vkVSCC7xljjrAI4UGfnKEQ", "bv2nCi5Qv5vroFiqKGopiw", "AEx2SYEUJmTxVVB18LlCwA", "5", "2016-05-28",
                     "Super simple place but amazing nonetheless",
                     "0", "0", "0"]
            , ["n6QzIUObkYshz4dz2QRJTw", "bv2nCi5Qv5vroFiqKGopiw", "VR6GpWIda3SfvPC-lg9H3w", "1", "2016-05-28",
               "Call for a reservation \n hi",
               "0", "0", "0"]
            , ["MV3CcKScW05u5LVfF6ok0g", "bv2nCi5Qv5vroFiqKGopiw", "CKC0-MOWMqoeWf6s-szl8g", "3", "2016-05-28",
               "Lester's is located in a beautiful",
               "0", "0", "0"]]

        # Act
        sut.format(iterator, 3, 5, output, max_process=2)

        # Assert
        output.seek(0)
        actual_lines =  output.readlines()
        self.assertEqual(len(actual_lines), len(iterator))
