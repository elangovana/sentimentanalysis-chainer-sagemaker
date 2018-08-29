class VocabFilter:
    def __init__(self, tokens_count_dict, min_frequency=2, max_vocab_size=10000):
        self.max_vocab_size = max_vocab_size
        self.tokens_count_dict = tokens_count_dict
        self.min_frequency = min_frequency
        self.top_words_vocab = None

    def __call__(self, word):
        top_words_vocab = self.top_words_vocab
        return not (word in top_words_vocab)

    @property
    def top_words_vocab(self):
        # lazy load, obtain topwords vocab on first load
        if self.__top_words_vocab__ is None:
            self.__top_words_vocab__ = {}
            for w, c in sorted(self.tokens_count_dict.items(), key=lambda x: (-x[1], x[0])):
                if len(self.__top_words_vocab__) >= self.max_vocab_size or c < self.min_frequency:
                    break
                self.__top_words_vocab__[w] = len(self.__top_words_vocab__)

        return self.__top_words_vocab__

    @top_words_vocab.setter
    def top_words_vocab(self, value):
        self.__top_words_vocab__ = value
