from io import StringIO
import random
from unittest import TestCase
import numpy as np

from encoders.PretrainedEmbedder import PretrainedEmbedder
from encoders.PretrainedEmbedderLoader import PretrainedEmbedderLoader


class TestPretrainedEmbedder(TestCase):

    def test_should_return_embeddings(self):
        ## Arrange
        embed_len = 10
        no_words = 12
        words_array = ["random_word_{}".format(r) for r in range(0, no_words)]
        other_words_embed = None
        emdedding = np.random.randint(0, 100, size=(no_words, embed_len))
        get_mock_embed_handle = self.get_mock_embedding_handle(words_array, emdedding)

        # Act
        sut = PretrainedEmbedderLoader()
        actual_word_index, actual_weights = sut(handle=get_mock_embed_handle,
                                                other_words_embed_dict=other_words_embed)

        # Assert
        for w, e in zip(words_array, emdedding):
            index = actual_word_index[w]
            self.assertSequenceEqual(actual_weights[index].tolist(), e.tolist())

    def test_should_return_embeddings_with_filter(self):
        ## Arrange
        prefix = "random_word_"
        embed_len = 10
        no_words = 12
        words_array = ["{}{}".format(prefix,r) for r in range(0, no_words)]
        other_words_embed = self.get_other_words_embed(embed_len, no_words)
        emdedding = np.random.randint(0, 100, size=(no_words, embed_len))
        get_mock_embed_handle = self.get_mock_embedding_handle(words_array, emdedding)

        # Act filter all words
        sut = PretrainedEmbedderLoader()
        actual_word_index, actual_weights = sut(handle=get_mock_embed_handle,
                                                other_words_embed_dict=other_words_embed, filter=lambda x: x.startswith(prefix))

        # Assert
        self.assertEqual(len(actual_weights), no_words)



    def test_should_return_embeddings_unknown(self):
        ## Arrange
        embed_len = 10
        no_words = 12
        words_array = ["random_word_{}".format(r) for r in range(0, no_words)]
        other_words_embed = self.get_other_words_embed(embed_len, no_words)
        emdedding = np.random.randint(0, 100, size=(no_words, embed_len))
        get_mock_embed_handle = self.get_mock_embedding_handle(words_array, emdedding)

        # Act
        sut = PretrainedEmbedderLoader()
        actual_word_index, actual_weights = sut(handle=get_mock_embed_handle, other_words_embed_dict=other_words_embed)

        # Assert
        # Assert0
        for w, e in other_words_embed.items():
            index = actual_word_index[w]
            self.assertSequenceEqual(actual_weights[index].tolist(), e.tolist())

    def get_other_words_embed(self, embed_len, no_words):
        # Other word embedding
        other_words_embed = {}

        for r in range(0, no_words):
            w = "other_word_{}".format(r)

            other_words_embed[w] = np.random.randint(0, 100, size=embed_len)
        return  other_words_embed

    def get_mock_embedding_handle(self, words, embed):
        stringIO = StringIO()

        for w, e in zip(words, embed):
            stringIO.write("{} {}\n".format(w, ' '.join(e.astype(str).tolist())))

        stringIO.seek(0)

        return stringIO
