import logging

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
        self.logger.info("The vocab size is {} and the nunits is {}".format(n_vocab, n_units))
        super(CNNEncoder, self).__init__()
        with self.init_scope():
            self.embed = L.EmbedID(n_vocab, n_units, ignore_label=-1,
                                   initialW=embed_init)
            self.cnn_w3 = L.Convolution2D(
                1, out_units, ksize=(2, n_units), stride=(1, 1), pad=(2 // 2, 0),
                nobias=True)
            self.cnn_w4 = L.Convolution2D(
                1, out_units, ksize=(3, n_units), stride=(1, 1), pad=(3 // 2, 0),
                nobias=True)

            self.mlp = MLP(n_layers, 3 * 3, dropout)

        self.out_units = 9
        self.dropout = dropout

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def __call__(self, xs):
        self.logger.debug("The length of the batch is {}".format(len(xs)))

        ## Concat the samples in the batch so they are are the same size, for shorter sentences, use -1 to indicate no word
        x_block = chainer.dataset.convert.concat_examples(xs, padding=-1)
        self.logger.debug("The shape  of the concatenated batch is {}".format(x_block.shape))
        self.logger.debug("The  block shape [0] of the concatenated set is {}".format(x_block[0].shape))

        ex_block = block_embed(self.embed, x_block, self.dropout)
        self.logger.debug("The embedded block shape of the concatenated set is {}".format(x_block.shape))
        self.logger.debug("The first embedded data shape is {}".format(ex_block[0].shape))

        h_w3 = F.max(self.cnn_w3(ex_block), axis=2)
        self.logger.debug("The first h_w3[0] data shape is {}".format(h_w3[0].shape))
        self.logger.debug("The first h_w3 data shape is {}".format(h_w3.shape))

        h_w4 = F.max(self.cnn_w4(ex_block), axis=2)
        h = F.concat([h_w3, h_w4], axis=1)
        h = F.relu(h)
        h = F.dropout(h, ratio=self.dropout)
        h = self.mlp(h)
        return h
