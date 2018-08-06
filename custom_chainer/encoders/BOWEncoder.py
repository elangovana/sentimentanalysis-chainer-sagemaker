import chainer
import numpy
from chainer import links as L, functions as F

from encoders.encoder_constants import embed_init
from encoders.encoder_helpers import block_embed


class BOWEncoder(chainer.Chain):

    """A BoW encoder with word embedding.

    This model encodes a sentence as just a set of words by averaging.

    Args:
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_vocab, n_units, dropout=0.1):
        super(BOWEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, ignore_label=-1,
                                   initialW=embed_init)

        self.out_units = n_units
        self.dropout = dropout

    def __call__(self, xs):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block)
        x_len = self.xp.array([len(x) for x in xs], numpy.int32)[:, None, None]
        h = F.sum(ex_block, axis=2) / x_len
        return h