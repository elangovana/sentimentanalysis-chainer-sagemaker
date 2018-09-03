import datetime
import json

import os

import chainer

import TextClassifier
from Train import Train
from VocabularyBuilder import VocabularyBuilder
from datasetyelp.YelpReviewDatasetProcessor import YelpReviewDatasetProcessor

from encoders.BOWNLPEncoder import BOWMLPEncoder
from encoders.CNNEncoder import CNNEncoder
from encoders.PretrainedEmbedder import PretrainedEmbedder
from encoders.RNNEncoder import RNNEncoder
from utils.NlpUtils import get_counts_by_token, UNKNOWN_WORD, EOS, transform_to_array
from utils.VocabFilter import VocabFilter


class TrainPipelineBuilder:
    def __init__(self, data_has_header=True, batchsize=64, char_based=False, dropout=.5, epoch=10, gpus=None,
                 no_layers=1,
                 embed_dim=300, embedding_file=None, max_vocab_size=20000, min_word_frequency=5,
                 best_model_snapshot_name='best_model.npz'):
        self.best_model_snapshot_name = best_model_snapshot_name
        self.data_has_header = data_has_header
        self.min_word_frequency = min_word_frequency
        self.max_vocab_size = max_vocab_size
        self.embedding_file = embedding_file
        self.embed_dim = embed_dim
        self.no_layers = no_layers
        self.gpus = gpus
        self.epoch = epoch
        self.dropout = dropout
        self.char_based = char_based
        self.batchsize = batchsize
        self.data_processor = None
        self.text_classifier = None
        self.vocab_builder = None
        self.trainer = None
        self.vocab_filter = None

    @property
    def vocab_builder(self):
        self.__vocab_builder__ = self.__vocab_builder__ or VocabularyBuilder(max_vocab_size=self.max_vocab_size,
                                                                             min_word_frequency=self.min_word_frequency)
        return self.__vocab_builder__

    @vocab_builder.setter
    def vocab_builder(self, value):
        self.__vocab_builder__ = value

    @property
    def text_classifier(self):
        self.__text_classifier__ = self.__text_classifier__ or TextClassifier.TextClassifier
        return self.__text_classifier__

    @text_classifier.setter
    def text_classifier(self, value):
        self.__text_classifier__ = value

    @property
    def vocab_filter(self):
        self.__vocab_filter__ = self.__vocab_filter__ or VocabFilter
        return self.__vocab_filter__

    @vocab_filter.setter
    def vocab_filter(self, value):
        self.__vocab_filter__ = value

    @property
    def trainer(self):
        self.__trainer__ = self.__trainer__ or Train
        return self.__trainer__

    @trainer.setter
    def trainer(self, value):
        self.__trainer__ = value

    def run(self, dataset_iterator, n_class, encoder_name, output_dir, validationset_iterator=None):
        #Split data set if no validation set
        if validationset_iterator is None:
            dataset_iterator, validationset_iterator = chainer.datasets.split_dataset_random(dataset_iterator, int(
                len(dataset_iterator) * 0.7) + 1, seed=777)

        word_count_dict = get_counts_by_token(dataset_iterator)

        filter = self.vocab_filter(word_count_dict, max_vocab_size=self.max_vocab_size,
                                   min_frequency=self.min_word_frequency,
                                   priority_words={UNKNOWN_WORD, EOS})

        weights, vocab = self.vocab_builder(dataset_iterator, self.embedding_file, embed_dim=self.embed_dim,
                                            vocab_filter=filter)

        # Setup a model

        encoder = self._get_encoder(encoder_name, vocab, weights, self.no_layers, self.embed_dim, self.dropout)

        model = self.text_classifier(encoder, n_class)

        train = self.trainer(encoder=encoder, vocab=vocab, out_dir=output_dir,
                             epoch=self.epoch, batchsize=self.batchsize, gpus=self.gpus)

        self.persist(output_dir, encoder_name, n_class, vocab, weights)

        train_data = transform_to_array(dataset_iterator, vocab, with_label=True)
        test_data = transform_to_array(validationset_iterator, vocab, with_label=True)

        train(train_data, test_data, model, self.best_model_snapshot_name)

    def persist(self, out_path, encoder_name, n_class, vocab, weights):
        if not os.path.isdir(out_path):
            os.mkdir(out_path)

        current_datetime = '{}'.format(datetime.datetime.today())
        model_setup = {}

        # persist weights
        weights_path = os.path.join(out_path, 'weights.json')
        with open(weights_path, 'w') as f:
            json.dump(weights.tolist(), f)

        # persist vocab
        vocab_path = os.path.join(out_path, 'vocab.json')
        with open(vocab_path, 'w') as f:
            json.dump(vocab, f)

        model_path = os.path.join(out_path, self.best_model_snapshot_name)
        model_setup['encoder_name'] = encoder_name
        model_setup['vocab_path'] = vocab_path
        model_setup['model_path'] = model_path
        model_setup['weights_path'] = weights_path
        model_setup['n_class'] = n_class
        model_setup['datetime'] = current_datetime
        model_setup['datetime'] = current_datetime
        model_setup['dropout'] = self.dropout
        model_setup['embed_dim'] = self.embed_dim
        model_setup['no_of_layers'] = self.no_layers
        with open(os.path.join(out_path, 'args.json'), 'w') as f:
            json.dump(model_setup, f)

    @staticmethod
    def load(model_dir):
        with open(os.path.join(model_dir, "args.json")) as args_file:
            setup_json = json.load(args_file)

        vocab_path = os.path.join(model_dir, os.path.basename(setup_json["vocab_path"]))
        model_path = os.path.join(model_dir, os.path.basename(setup_json["model_path"]))
        weights_path = os.path.join(model_dir, os.path.basename(setup_json["weights_path"]))

        # load vocab
        with open(vocab_path) as vocab_handle:
            vocab = json.load(vocab_handle)

        # load weights

        with open(weights_path) as weights_handle:
            weights = json.load(weights_handle)

        n_class = setup_json['n_class']
        dropout = setup_json['dropout']
        embed_dim = setup_json['embed_dim']
        no_layers = setup_json['no_of_layers']
        # Setup a model
        encoder = TrainPipelineBuilder._get_encoder(setup_json['encoder_name'], vocab, weights, no_layers, embed_dim,
                                                    dropout)

        model = TextClassifier.TextClassifier(encoder, n_class)
        chainer.serializers.load_npz(model_path, model)

        return (model, vocab, setup_json)

    @staticmethod
    def _get_encoder(encoder_name, vocab, weights, no_layers, embed_dim, dropout):
        # TODO: Make Embedder pluggable
        embedder = PretrainedEmbedder(vocab, weights)

        if encoder_name == 'rnn':
            encoder = RNNEncoder(n_layers=no_layers, n_vocab=len(vocab), n_units=embed_dim,
                                 dropout=dropout,
                                 embedder=embedder)
        elif encoder_name == 'bow':
            encoder = BOWMLPEncoder(n_layers=no_layers, n_vocab=len(vocab), n_units=embed_dim,
                                    dropout=dropout)
        elif encoder_name == "cnn":
            encoder = CNNEncoder(n_layers=no_layers, n_vocab=len(vocab), n_units=embed_dim,
                                 dropout=dropout,
                                 embedder=embedder)
        else:
            raise ValueError("The model name {} is unknown. Please provide a valid model name".format(encoder_name))
        return encoder
