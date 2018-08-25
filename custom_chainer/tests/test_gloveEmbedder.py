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
        i = [1, 2]

        ## Act
        sut = GloveEmbedder(get_mock_embed_handle, other_words_embed)
        actual = sut(i)

        # Assert
        self.assertListEqual(actual.tolist(), emdedding[i].tolist())

    def test_should_return_embeddings_unknown(self):
        ## Arrange
        embed_len = 10
        no_words = 12
        words_array = ["random_word_{}".format(r) for r in range(0, no_words)]
        other_words_embed = self.get_other_words_embed(embed_len, no_words)
        emdedding = np.random.randint(0, 100, size=(no_words, embed_len))
        get_mock_embed_handle = self.get_mock_embedding_handle(words_array, emdedding)
        word, expected_embed = random.choice(list(other_words_embed.items()))

        ## Act
        sut = GloveEmbedder(get_mock_embed_handle, other_words_embed)
        i = sut.word_index[word]
        actual = sut([i])

        # Assert

        self.assertListEqual(actual[0].tolist(), expected_embed.tolist())

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
