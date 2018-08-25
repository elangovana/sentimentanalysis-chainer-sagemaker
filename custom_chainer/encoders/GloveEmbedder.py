import chainer
import numpy as np
from chainer.backends import cuda


class GloveEmbedder:

    def __init__(self, handle):
        self.word_index, self.weights = self.load(handle)
        self.__weights__ = None

    def __call__(self, array):
        return self.weights[array]



    def load(self, handle):
        """
Expects the stream to string to contain embedding in space separated format with the first column containing the word itself.
Each record is separated by new lines
e.g for an embedding of size 10
zsombor -0.75898 -0.47426 0.4737 0.7725 -0.78064 0.23233 0.046114 0.84014 0.243710 .022978
sandberger 0.072617 -0.51393 0.4728 -0.52202 -0.35534 0.34629 0.23211 0.23096 0.26694 .41028
        :param handle: handle containing the embedding
        """
        word_index_dict = {}
        embeddings_array = []
        index = 0
        xp = cuda.get_array_module()
        for line in handle:
            values = line.split()
            word = values[0]
            embeddings = xp.asarray(values[1:], dtype='float32')
            word_index_dict[word] = index
            embeddings_array.append(embeddings)

        embeddings_array = xp.asarray(embeddings_array, dtype='float32')
        return word_index_dict, embeddings_array
