import logging

import chainer
import numpy as np
from chainer.backends import cuda


class PretrainedEmbedderLoader:

    def __call__(self, handle, other_words_embed_dict, filter=None):
        """
Expects the stream of strings to contain word embedding. Each record must be in space separated format with the first column containing the word itself.
Each record is separated by new lines
e.g for an embedding of size 10
zsombor -0.75898 -0.47426 0.4737 0.7725 -0.78064 0.23233 0.046114 0.84014 0.243710 .022978
sandberger 0.072617 -0.51393 0.4728 -0.52202 -0.35534 0.34629 0.23211 0.23096 0.26694 .41028
        :param handle: handle containing the embedding
        """
        word_index_dict = {}
        embeddings_array = []
        other_words_embed_dict = other_words_embed_dict or {}
        xp = cuda.get_array_module()

        # Load embeddings from file
        for line in handle:
            values = line.split()
            word = values[0]

            # if word needs to be filtered, ignore word
            if filter is not None and filter(word): continue

            # Not ignored word
            embeddings = xp.asarray(values[1:], dtype='float32')
            word_index_dict[word] = len(word_index_dict)
            embeddings_array.append(embeddings)

        # Add  embeddings for additional words that do not exist in handle
        for w in other_words_embed_dict.keys():
            # if word needs to be filtered, ignore word
            if filter is not None and filter(w): continue

            # Not ignored word
            if word_index_dict.get(w, None) is None:
                word_index_dict[w] = len(word_index_dict)
                embeddings_array.append(xp.asarray(other_words_embed_dict[w], dtype='float32'))

        # Convert to ndarray or cupy
        embeddings_array = xp.asarray(embeddings_array, dtype='float32')
        return word_index_dict, embeddings_array
