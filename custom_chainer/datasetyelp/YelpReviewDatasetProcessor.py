import logging

import chainer

from utils.NlpUtils import split_text, normalize_text, make_vocab


# TODO: Clean up this class
class YelpReviewDatasetProcessor(chainer.dataset.iterator.Iterator):

    def __init__(self, iterator, vocab=None, seed=777):
        super(YelpReviewDatasetProcessor, self).__init__()
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
        tokens = self.extract_tokens(self.get_review_text(record))
        # construct array, so use word index from vocab
        #tokens_index = make_array(tokens, self.vocab)
        result = (tokens)
        if self.has_label(record):
            label = self.get_label(record)
            result = (tokens, label)
        return result

    def __len__(self):
        return len(self.iterator)

    def extract_tokens(self, review_text):
        tokens = split_text(normalize_text(review_text), False)
        return tokens

    def has_label(self, record):
        # Assumes has label if the record has all the fields
        return len(record) == 9

    @staticmethod
    def get_label(record):
        assert len(record) == 9

        stars = int(record[3])
        label = YelpReviewDatasetProcessor.convert_rating_to_sentiment(stars)

        return label

    @staticmethod
    def convert_rating_to_sentiment(stars):
        label = 0
        if stars > 3:
            label = 1
        elif label < 3:
            label = -1
        return label

    def __iter__(self):
        return self.items()

    def items(self):
        for i in range(0, len(self)):
            yield self[i]

    @staticmethod
    def get_review_text(record):
        if len(record) == 9:
            return record[5]
        if len(record) == 1:
            return record[0]

    def construct_vocab(self):
        tokens_array = [[self.extract_tokens(self.get_review_text(record))] for record in self.iterator]
        return make_vocab(tokens_array, tokens_index=0)





