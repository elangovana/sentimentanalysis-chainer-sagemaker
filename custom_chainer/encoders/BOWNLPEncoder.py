import chainer

from encoders.MLP import MLP
from encoders.BOWEncoder import BOWEncoder


class BOWMLPEncoder(chainer.Chain):

    """A BOW encoder with word embedding and MLP.

    This model encodes a sentence as just a set of words by averaging.
    Additionally, its output is fed into a multilayer perceptron.

    Args:
        n_layers (int): The number of layers of MLP.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of MLP and word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        super(BOWMLPEncoder, self).__init__()
        with self.init_scope():
            self.bow_encoder = BOWEncoder(n_vocab, n_units, dropout)
            self.mlp_encoder = MLP(n_layers, n_units, dropout)

        self.out_units = n_units

    def __call__(self, xs):
        h = self.bow_encoder(xs)
        h = self.mlp_encoder(h)
        return h