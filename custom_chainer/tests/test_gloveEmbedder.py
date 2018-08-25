from io import StringIO
import random
from unittest import TestCase
import numpy as np

from encoders.GloveEmbedder import GloveEmbedder


class TestGloveEmbedder(TestCase):

    def test_should_return_embeddings(self):
        ## Arrange
        embed_len = 10
        no_words = 12
        words_array = ["random_word_{}".format(r) for r in range(0, no_words)]
        other_words_embed = None
        emdedding = np.random.randint(0, 100, size=(no_words, embed_len))
        get_mock_embed_handle = self.get_mock_embedding_handle(words_array, emdedding)
        actual = np.random.randint(0, 100, size=(no_words, embed_len))

        # Act
        sut = GloveEmbedder(get_mock_embed_handle, other_words_embed)
        sut(actual)

        # Assert
        self.assertListEqual(actual.tolist()[0:no_words], emdedding.tolist())

    def test_should_return_embeddings_unknown(self):
        ## Arrange
        embed_len = 10
        no_words = 12
        words_array = ["random_word_{}".format(r) for r in range(0, no_words)]
        other_words_embed = self.get_other_words_embed(embed_len, no_words)
        emdedding = np.random.randint(0, 100, size=(no_words, embed_len))
        get_mock_embed_handle = self.get_mock_embedding_handle(words_array, emdedding)
        actual = np.random.randint(0, 100, size=(no_words * 2, embed_len))

        # Act
        sut = GloveEmbedder(get_mock_embed_handle, other_words_embed)
        sut(actual)

        # Assert
        expected = [[0] for i in range(0, no_words)]
        for w in other_words_embed.keys():
            i = sut.word_index[w] - no_words
            expected[i]=other_words_embed[w].tolist()
        self.assertListEqual(actual.tolist()[no_words:2*no_words], expected)

    def get_other_words_embed(self, embed_len, no_words):
        # Other word embedding
        other_words_embed = {}
        for r in range(0, no_words):
            w = "other_word_{}".format(r)
            other_words_embed[w] = np.random.randint(0, 100, size=embed_len)
        return other_words_embed

    def get_mock_embedding_handle(self, words, embed):
        stringIO = StringIO()

        for w, e in zip(words, embed):
            stringIO.write("{} {}\n".format(w, ' '.join(e.astype(str).tolist())))

        stringIO.seek(0)

        return stringIO
