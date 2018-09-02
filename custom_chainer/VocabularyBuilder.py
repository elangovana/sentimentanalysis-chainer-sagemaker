import logging
import numpy as np
from encoders.PretrainedEmbedderLoader import PretrainedEmbedderLoader
from utils.NlpUtils import get_counts_by_token, make_vocab, UNKNOWN_WORD, EOS


class VocabularyBuilder:
    def __init__(self, max_vocab_size, min_word_frequency):
        self.min_word_frequency = min_word_frequency
        self.max_vocab_size = max_vocab_size

        self.embedder_loader = None

    @property
    def logger(self):
        return logging.getLogger(__name__)

    @property
    def embedder_loader(self):
        self.__embedder_loader__ = self.__embedder_loader__ or PretrainedEmbedderLoader()
        return self.__embedder_loader__

    @embedder_loader.setter
    def embedder_loader(self, value):
        self.__embedder_loader__ = value

    def __call__(self, dataset, embedding_file=None, embed_dim=300, vocab_filter=None):
        word_count_dict = get_counts_by_token(dataset, tokens_index=0)
        self.logger.info('Total tokens found in data: {}'.format(len(word_count_dict)))

        weights = None
        if embedding_file is not None:
            weights, vocab = self._get_vocab_weights(embedding_file, embed_dim, vocab_filter=vocab_filter)
        else:
            vocab = make_vocab(dataset, max_vocab_size=self.max_vocab_size, min_freq=self.min_word_frequency,
                               tokens_index=0)
            weights = np.random.uniform(-0.5, .5, size=(len(vocab), embed_dim))

        return weights, vocab

    def _get_vocab_weights(self, embedding_file, embed_dim, vocab_filter):
        weights = None
        vocab = None
        unknown_words_rand = np.random.uniform(-0.5, .5, size=(2, embed_dim))
        if embedding_file is None:  return weights, vocab

        # load embeddings from file
        with open(embedding_file, encoding='utf-8') as f:
            word_index, weights = self.embedder_loader(f, {UNKNOWN_WORD: unknown_words_rand[0], EOS: unknown_words_rand[1]},
                                                       vocab_filter)
            vocab = word_index

        return weights, vocab
