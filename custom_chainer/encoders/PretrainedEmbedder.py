import logging

import chainer
import numpy as np
from chainer.backends import cuda


class PretrainedEmbedder:

    def __init__(self, word_index, weights):

        self.word_index, self.weights = word_index, weights
        self.__weights__ = None

    def __call__(self, array):
        self.logger.info("Loading pretrainined embeddings...")
        xp = cuda.get_array_module(array)
        array[...] = xp.asarray(self.weights)

    @property
    def logger(self):
        return logging.getLogger(__name__)
