import csv
import io
import logging

import chainer
import numpy

from NlpUtils import split_text, normalize_text, make_vocab, transform_to_array, make_array


class MovieDatasetIteratorProcessor(chainer.dataset.iterator.Iterator):
    def __init__(self, iterator, vocab=None, seed=777):
        super(MovieDatasetIteratorProcessor, self).__init__()
        self.seed = seed
        self.vocab = vocab
        self.iterator = iterator

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __getitem__(self, idx):
        # reporting progress...
        line = self.iterator[idx]
        tokens = self.extract_tokens(line[0])
        tokens_index = make_array(tokens, self.vocab)
        result = (tokens_index)
        if self.has_label(line):
            result = (tokens_index, line[1])
        return result

    def __len__(self):
        return len(self.iterator)

    def extract_tokens(self, review_text):
        tokens = split_text(normalize_text(review_text), False)
        return tokens

    def has_label(self, line):
        return len(line) == 2
