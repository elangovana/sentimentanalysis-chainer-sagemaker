import chainer
from chainer import links as L, functions as F


class MLP(chainer.ChainList):

    """A multilayer perceptron.

    Args:
        n_vocab (int): The size of vocabulary.
        n_units (int): The number of units in a hidden or output layer.
        dropout (float): The dropout ratio.

    """

    def __init__(self, n_layers, n_units, dropout=0.1):
        super(MLP, self).__init__()
        for i in range(n_layers):
            self.add_link(L.Linear(None, n_units))
        self.dropout = dropout
        self.out_units = n_units

    def __call__(self, x):
        for i, link in enumerate(self.children()):
            x = F.dropout(x, ratio=self.dropout)
            x = F.relu(link(x))
        return x