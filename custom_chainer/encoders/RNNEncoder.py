import chainer
from chainer import links as L

from encoders.encoder_helpers import sequence_embed
from encoders.encoder_constants import embed_init


class RNNEncoder(chainer.Chain):

    """A LSTM-RNN Encoder with Word Embedding.

    This model encodes a sentence sequentially using LSTM.

    Args:
        n_layers (int): The number of LSTM layers.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of a LSTM layer and word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        super(RNNEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units,
                                   initialW=embed_init)
            self.encoder = L.NStepLSTM(n_layers, n_units, n_units, dropout)

        self.n_layers = n_layers
        self.out_units = n_units
        self.dropout = dropout

    def __call__(self, xs):
        exs = sequence_embed(self.embed, xs, self.dropout)
        last_h, last_c, ys = self.encoder(None, None, exs)
        assert(last_h.shape == (self.n_layers, len(xs), self.out_units))
        concat_outputs = last_h[-1]
        return concat_outputs