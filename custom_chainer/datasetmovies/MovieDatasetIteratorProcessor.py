import logging

import chainer

from utils.NlpUtils import split_text, normalize_text, make_vocab


class MovieDatasetIteratorProcessor(chainer.dataset.iterator.Iterator):
    def __init__(self, iterator, vocab=None, seed=777):
        super(MovieDatasetIteratorProcessor, self).__init__()
        self.seed = seed
        self.vocab = vocab
        self.iterator = iterator

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def vocab(self):
        self.__vocab__ = self.__vocab__ or self.construct_vocab()
        return self.__vocab__

    @vocab.setter
    def vocab(self, value):
        self.__vocab__ = value

    def __getitem__(self, idx):
        record = self.iterator[idx]
        self.logger.debug("Reading index {}, record {}".format(idx, record))
        tokens = self.extract_tokens(record[0])
        self.logger.debug("Parsed into tokens {}".format(tokens))
        result = (tokens)
        if self.has_label(record):
            result = (tokens, record[1])
        return result

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        return self.items()

    def items(self):
        for i in range(0, len(self)):
            yield self[i]

    def extract_tokens(self, review_text):
        tokens = split_text(normalize_text(review_text), False)
        return tokens

    def has_label(self, line):
        return len(line) == 2

    def construct_vocab(self):
        tokens_array = [[self.extract_tokens(record[0])] for record in self.iterator]
        return make_vocab(tokens_array, tokens_index=0)
