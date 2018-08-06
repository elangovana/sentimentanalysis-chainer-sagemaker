import chainer
from chainer import links as L, functions as F

from encoders.EncoderConstants import embed_init
from encoders.EncoderHelpers import block_embed
from encoders.MLP import MLP


class CNNEncoder(chainer.Chain):

    """A CNN encoder with word embedding.

    This model encodes a sentence as a set of n-gram chunks
    using convolutional filters.
    Following the convolution, max-pooling is applied over time.
    Finally, the output is fed into a multilayer perceptron.

    Args:
        n_layers (int): The number of layers of MLP.
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units of MLP and word embedding.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_vocab, n_units, dropout=0.1):
        out_units = n_units // 3
        super(CNNEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, ignore_label=-1,
                                   initialW=embed_init)
            self.cnn_w3 = L.Convolution2D(
                n_units, out_units, ksize=(3, 1), stride=1, pad=(2, 0),
                nobias=True)
            self.cnn_w4 = L.Convolution2D(
                n_units, out_units, ksize=(4, 1), stride=1, pad=(3, 0),
                nobias=True)
            self.cnn_w5 = L.Convolution2D(
                n_units, out_units, ksize=(5, 1), stride=1, pad=(4, 0),
                nobias=True)
            self.mlp = MLP(n_layers, out_units * 3, dropout)

        self.out_units = out_units * 3
        self.dropout = dropout

    def __call__(self, xs):
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        ex_block = block_embed(self.embed, x_block, self.dropout)
        h_w3 = F.max(self.cnn_w3(ex_block), axis=2)
        h_w4 = F.max(self.cnn_w4(ex_block), axis=2)
        h_w5 = F.max(self.cnn_w5(ex_block), axis=2)
        h = F.concat([h_w3, h_w4, h_w5], axis=1)
        h = F.relu(h)
        h = F.dropout(h, ratio=self.dropout)
        h = self.mlp(h)
        return h