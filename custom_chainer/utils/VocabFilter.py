class VocabFilter:
    def __init__(self, tokens_count_dict, min_frequency=2):
        self.tokens_count_dict = tokens_count_dict
        self.min_frequency = min_frequency

    def __call__(self, vocab):
        vocab_filtered = {}
        filtered_tokens_count = filter(lambda k: k[1] >= self.min_frequency, self.tokens_count_dict.items())
        for k, v in filtered_tokens_count:
            if k in vocab.keys():
                vocab_filtered[k] = vocab[k]

        return vocab_filtered
