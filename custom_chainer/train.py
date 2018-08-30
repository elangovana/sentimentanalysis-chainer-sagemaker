import datetime
import json
import logging
import os

import chainer
import numpy as np
from chainer import training
from chainer.training import extensions

import TextClassifier
from encoders.PretrainedEmbedderLoader import PretrainedEmbedderLoader
from utils.NlpUtils import UNKNOWN_WORD, EOS, make_vocab, get_counts_by_token

from encoders.BOWNLPEncoder import BOWMLPEncoder
from encoders.CNNEncoder import CNNEncoder
from encoders.PretrainedEmbedder import PretrainedEmbedder
from encoders.RNNEncoder import RNNEncoder
from gpu_utils import convert_seq
from dataprep.YelpReviewDatasetProcessor import YelpReviewDatasetProcessor
from utils.VocabFilter import VocabFilter


def run_train(batchsize, char_based, dataset, dropout, epoch, max_gpu_id, model, no_layers, out,
              unit, embedding_file=None, shuffle=False, max_vocab_size=20000, min_word_frequency=10):
    # Has to be the first line so that the args can be persisted
    current_args = locals()
    current_datetime = '{}'.format(datetime.datetime.today())

    logger = logging.getLogger(__name__)

    data_processor = YelpReviewDatasetProcessor()

    train, test = data_processor.parse(
        dataset, char_based=char_based)

    vocab = make_vocab(train, max_vocab_size=max_vocab_size, min_freq=min_word_frequency, tokens_index=0)
    word_count_dict = get_counts_by_token(train, tokens_index=0)
    embbedder = None
    if embedding_file is not None:
        vocabfilter = VocabFilter(word_count_dict, max_vocab_size=max_vocab_size, min_frequency=min_word_frequency)
        embbedder, vocab = get_embedder(embedding_file, unit, filter=vocabfilter)

    test, train, vocab = data_processor.one_hot_encode(train, test, vocab)

    logger.info('# train data: {}'.format(len(train)))
    logger.info('# test  data: {}'.format(len(test)))
    logger.info('# vocab: {}'.format(len(vocab)))
    n_classes = [int(d[1]) for d in train]
    unique_classes, counts_classes = np.unique(n_classes, return_counts=True)
    logger.info(
        "Class distribution: \n{}".format(np.asarray((unique_classes, counts_classes))))

    n_class = len(unique_classes)

    train_iter = chainer.iterators.SerialIterator(train, batchsize, shuffle=shuffle)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)
    # Setup a model
    Encoder = CNNEncoder
    if model == 'rnn':
        Encoder = RNNEncoder
    # elif model == 'bow':
    #     Encoder = BOWMLPEncoder

    encoder = Encoder(n_layers=no_layers, n_vocab=len(vocab), n_units=unit, dropout=dropout, embedder=embbedder)

    model = TextClassifier.TextClassifier(encoder, n_class)

    # Check if can distribute training
    use_distibuted_gpu = False
    use_gpu = False
    main_device = -1

    if max_gpu_id >= 0:
        main_device = 0
        use_gpu = True
        use_distibuted_gpu = max_gpu_id > 1

    if use_gpu and not use_distibuted_gpu:
        chainer.backends.cuda.get_device_from_id(main_device).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    # Set up a trainer
    if use_distibuted_gpu:
        device_dict = {}
        for device_id in range(0, max_gpu_id + 1):
            if device_id == main_device:
                device_dict['main'] = device_id
            else:
                device_dict[str(device_id)] = device_id

        # ParallelUpdater implements the data-parallel gradient computation on
        # multiple GPUs. It accepts "devices" argument that specifies which GPU to
        # use.
        updater = training.updaters.ParallelUpdater(
            train_iter,
            optimizer,
            converter=convert_seq,
            # The device of the name 'main' is used as a "master", while others are
            # used as slaves. Names other than 'main' are arbitrary.
            devices=device_dict,
        )

    else:
        updater = training.updaters.StandardUpdater(
            train_iter, optimizer,
            converter=convert_seq, device=main_device)

    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)
    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(
        test_iter, model,
        converter=convert_seq, device=main_device))
    # Take a best snapshot
    record_trigger = training.triggers.MaxValueTrigger(
        'validation/main/accuracy', (1, 'epoch'))
    trainer.extend(extensions.snapshot_object(
        model, 'best_model.npz'),
        trigger=record_trigger)
    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    # Print a progress bar to stdout
    # trainer.extend(extensions.ProgressBar())
    # Save vocabulary and model's setting
    persist(out, current_args, current_datetime, n_class, vocab)
    # Run the training
    trainer.run()


def persist(out_path, args_to_persist, current_datetime, n_class, vocab):
    if not os.path.isdir(out_path):
        os.mkdir(out_path)
    vocab_path = os.path.join(out_path, 'vocab.json')
    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    model_path = os.path.join(out_path, 'best_model.npz')
    model_setup = args_to_persist
    model_setup['vocab_path'] = vocab_path
    model_setup['model_path'] = model_path
    model_setup['n_class'] = n_class
    model_setup['datetime'] = current_datetime
    with open(os.path.join(out_path, 'args.json'), 'w') as f:
        json.dump(model_setup, f)


def get_embedder(embedding_file, unit, filter):
    embbedder = None
    vocab = None
    rand_embed = np.random.uniform(-0.5, .5, size=(2, unit))
    if (embedding_file is not None):
        with open(embedding_file, encoding='utf-8') as f:
            word_index, weights = PretrainedEmbedderLoader()(f, {UNKNOWN_WORD: rand_embed[0], EOS: rand_embed[1]},
                                                             filter)
            embbedder = PretrainedEmbedder(word_index, weights)
            vocab = embbedder.word_index
    return embbedder, vocab
