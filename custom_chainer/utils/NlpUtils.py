import collections
import io

import numpy

UNKNOWN_WORD = '<unk>'

EOS = '<eos>'


def split_text(text, char_based=False):
    if char_based:
        return list(text)
    else:
        return text.split()


def normalize_text(text):
    return text.strip().lower()


def make_vocab(dataset, max_vocab_size=20000, min_freq=2, tokens_index = 0):
    counts = get_counts_by_token(dataset, tokens_index)

    vocab = {EOS: 0, UNKNOWN_WORD: 1}
    for w, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        if len(vocab) >= max_vocab_size or c < min_freq:
            break
        vocab[w] = len(vocab)
    return vocab


def get_counts_by_token(dataset, tokens_index):
    counts = collections.defaultdict(int)
    for record in dataset:
        tokens = record[tokens_index]
        for token in tokens:
            counts[token] += 1
    return counts


def make_array(tokens, vocab, add_eos=True):
    unk_id = vocab[UNKNOWN_WORD]
    eos_id = vocab[EOS]
    ids = [vocab.get(token, unk_id) for token in tokens]
    if add_eos:
        ids.append(eos_id)
    return numpy.array(ids, numpy.int32)


def transform_to_array(dataset, vocab, with_label=True):
    if with_label:
        return [(make_array(tokens, vocab), numpy.array([cls], numpy.int32))
                for tokens, cls in dataset]
    else:
        return [make_array(tokens, vocab)
                for tokens in dataset]
