import chainer


class RandomEmbedder:

    def __init__(self):
        self.embedder = chainer.initializers.Uniform(.25)

    def __call__(self, *args):
        return self.embedder(*args)
