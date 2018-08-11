import datetime
import json
import logging
import os

import chainer
import numpy as np
from chainer import training
from chainer.training import extensions

import TextClassifier

from encoders.BOWNLPEncoder import BOWMLPEncoder
from encoders.CNNEncoder import CNNEncoder
from encoders.RNNEncoder import RNNEncoder
from gpu_utils import convert_seq
from dataprep.YelpReviewDatasetProcessor import YelpReviewDatasetProcessor


def run_train(batchsize, char_based, dataset, dropout, epoch, gpu, model, no_layers, out,
              unit):
    # Has to be the first line so that the args can be persisted
    current_args = locals()
    current_datetime = '{}'.format(datetime.datetime.today())

    logger = logging.getLogger(__name__)

    data_processor = YelpReviewDatasetProcessor()
    train, test, vocab = data_processor.get_dataset(
        dataset, char_based=char_based)
    logger.info('# train data: {}'.format(len(train)))
    logger.info('# test  data: {}'.format(len(test)))
    logger.info('# vocab: {}'.format(len(vocab)))
    n_classes = [int(d[1]) for d in train]
    unique_classes, counts_classes = np.unique(n_classes, return_counts=True)
    logger.info(
        "Class distribution: \n{}".format(np.asarray((unique_classes, counts_classes))))

    n_class = len(unique_classes)

    # TODO: Make shuffle as option..Shuffling is set to false because assuming that the splitter is already run, which shuffles the data.

    train_iter = chainer.iterators.SerialIterator(train, batchsize, shuffle=False)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)
    # Setup a model
    Encoder = CNNEncoder
    if model == 'rnn':
        Encoder = RNNEncoder
    elif model == 'bow':
        Encoder = BOWMLPEncoder
    encoder = Encoder(n_layers=no_layers, n_vocab=len(vocab),
                      n_units=unit, dropout=dropout)
    model = TextClassifier.TextClassifier(encoder, n_class)

    # Check if can distribute training
    use_distibuted_gpu = False
    use_gpu = False
    if gpu >= 0:
        use_gpu = True
        use_distibuted_gpu = gpu > 1

    if use_gpu and not use_distibuted_gpu:
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

    # Set up a trainer
    if use_distibuted_gpu:
        device_dict = {}
        for g in range(1, gpu + 1):
            device_dict[str(g)] = g
        device_dict['main'] = 0


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
            converter=convert_seq, device=gpu)

    trainer = training.Trainer(updater, (epoch, 'epoch'), out=out)
    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(
        test_iter, model,
        converter=convert_seq, device=gpu))
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
    if not os.path.isdir(out):
        os.mkdir(out)

    vocab_path = os.path.join(out, 'vocab.json')

    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    model_path = os.path.join(out, 'best_model.npz')
    model_setup = current_args
    model_setup['vocab_path'] = vocab_path
    model_setup['model_path'] = model_path
    model_setup['n_class'] = n_class
    model_setup['datetime'] = current_datetime
    with open(os.path.join(out, 'args.json'), 'w') as f:
        json.dump(model_setup, f)
    # Run the training
    trainer.run()
