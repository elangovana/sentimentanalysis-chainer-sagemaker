from io import StringIO
from unittest import TestCase
import numpy as np

from encoders.GloveEmbedder import GloveEmbedder


class TestGloveEmbedder(TestCase):

    def test_should_return_embeddings(self):
        ## Arrange
        embed_len = 10
        no_words = 12
        words_array = ["random_word_{}".format(r) for r in range(0, no_words)]
        emdedding = np.random.randint(0, 100, size=(no_words, embed_len))
        i = [1, 2]

        ## Act
        sut = GloveEmbedder(self.get_mock_embedding(words_array, emdedding))
        actual = sut(i)

        # Assert
        self.assertListEqual(actual.tolist(), emdedding[i].tolist())

    def get_mock_embedding(self, words, embed):
        stringIO = StringIO()

        for w, e in zip(words, embed):
            stringIO.write("{} {}\n".format(w, ' '.join(e.astype(str).tolist())))

        stringIO.seek(0)

        return stringIO
