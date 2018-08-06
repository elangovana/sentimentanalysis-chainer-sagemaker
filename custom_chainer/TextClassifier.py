import chainer
from chainer import links as L, functions as F, reporter


class TextClassifier(chainer.Chain):

    """A classifier using a given encoder.

     This chain encodes a sentence and classifies it into classes.

     Args:
         encoder (Link): A callable encoder, which extracts a feature.
             Input is a list of variables whose shapes are
             "(sentence_length, )".
             Output is a variable whose shape is "(batchsize, n_units)".
         n_class (int): The number of classes to be predicted.

     """

    def __init__(self, encoder, n_class, dropout=0.1):
        super(TextClassifier, self).__init__()
        with self.init_scope():
            self.encoder = encoder
            self.output = L.Linear(encoder.out_units, n_class)
        self.dropout = dropout

    def __call__(self, xs, ys):
        concat_outputs = self.predict(xs)
        concat_truths = F.concat(ys, axis=0)

        loss = F.softmax_cross_entropy(concat_outputs, concat_truths)
        accuracy = F.accuracy(concat_outputs, concat_truths)
        reporter.report({'loss': loss.data}, self)
        reporter.report({'accuracy': accuracy.data}, self)
        return loss

    def predict(self, xs, softmax=False, argmax=False):
        concat_encodings = F.dropout(self.encoder(xs), ratio=self.dropout)
        concat_outputs = self.output(concat_encodings)
        if softmax:
            return F.softmax(concat_outputs).data
        elif argmax:
            return self.xp.argmax(concat_outputs.data, axis=1)
        else:
            return concat_outputs