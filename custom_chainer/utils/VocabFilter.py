class VocabFilter:
    def __init__(self, tokens_count_dict: dict, min_frequency: int = 2, max_vocab_size: int = 10000,
                 filter_unknown_words=False):
        """
Filters word based on occurrence frequency. if the number of tokens is less than max_vocab_size, then the min frequency is ignored
        :param tokens_count_dict:
        :param min_frequency:
        :param max_vocab_size:
        """
        self.filter_unknown_words = filter_unknown_words
        self.max_vocab_size = max_vocab_size
        self.tokens_count_dict = tokens_count_dict
        self.min_frequency = min_frequency
        self.top_words_vocab = None

    def __call__(self, word):
        # check if word exists in vocab first before applying the filter
        if not (self.filter_unknown_words or not (word not in self.tokens_count_dict)):
            return False

        return not (word in self.top_words_vocab)

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
