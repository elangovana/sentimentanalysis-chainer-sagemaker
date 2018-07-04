#!/usr/bin/env python
import argparse
import datetime
import json
import os

import chainer
from chainer import training
from chainer.training import extensions

import nets
from nlp_utils import convert_seq
from yelp_review_dataset_processor import YelpReviewDatasetProcessor


def main():
    current_datetime = '{}'.format(datetime.datetime.today())


    parser = argparse.ArgumentParser(
        description='Chainer example: Text Classification')
    parser.add_argument('--traindata', required=True,
                        help='The dataset file with respect to the train directory')
    parser.add_argument('--traindata-dir',
                        help='The directory containing training artifacts such as training data', default=os.environ.get('SM_CHANNEL_TRAIN', "."))
    parser.add_argument('--testdata', '-td',
                        help='The test dataset file with respect to the test directory', default=None)
    parser.add_argument('--testdata-dir', '-tdir',
                        help='The directory containing test artifacts such as test data', default=os.environ.get('SM_CHANNEL_TEST', "."))
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=os.environ.get('SM_NUM_GPUS', 0)-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default=os.environ.get('SM_OUTPUT_DATA_DIR', "result_data"),
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='Number of layers of RNN or MLP following CNN')
    parser.add_argument('--dropout', '-d', type=float, default=0.4,
                        help='Dropout rate')

    parser.add_argument('--model', '-model', default='cnn',
                        choices=['cnn', 'rnn', 'bow'],
                        help='Name of encoder model type.')
    parser.add_argument('--char-based', action='store_true')

    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=2))


    #args parse

    args_dict = args.__dict__
    batchsize = args.batchsize
    model = args.model
    training_set= os.path.join(args.traindata_dir, args.traindata)
    validation_set = os.path.join(args.testdata_dir, args.testdata) if args.testdata is not None else None
    dataset = [training_set, validation_set] if validation_set is not None else [training_set]
    char_based = args.char_based
    no_layers = args.layer
    unit = args.unit
    dropout = args.dropout
    gpu = args.gpu
    epoch = args.epoch
    out = args.out

    run_train(args_dict, batchsize, char_based, current_datetime, dataset, dropout, epoch, gpu, model, no_layers, out,
              unit)


def run_train(args_dict, batchsize, char_based, current_datetime, dataset, dropout, epoch, gpu, model, no_layers, out,
              unit):
    # Load a dataset
    data_processor = YelpReviewDatasetProcessor()
    train, test, vocab = data_processor.get_dataset(
        dataset, char_based=char_based)
    print('# train data: {}'.format(len(train)))
    print('# test  data: {}'.format(len(test)))
    print('# vocab: {}'.format(len(vocab)))
    n_class = len(set([int(d[1]) for d in train]))
    print('# class: {}'.format(n_class))
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize,
                                                 repeat=False, shuffle=False)
    # Setup a model
    Encoder = nets.CNNEncoder
    if model == 'rnn':
        Encoder = nets.RNNEncoder
    elif model == 'bow':
        Encoder = nets.BOWMLPEncoder
    encoder = Encoder(n_layers=no_layers, n_vocab=len(vocab),
                      n_units=unit, dropout=dropout)
    model = nets.TextClassifier(encoder, n_class)
    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    # Setup an optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))
    # Set up a trainer
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
    trainer.extend(extensions.ProgressBar())
    # Save vocabulary and model's setting
    if not os.path.isdir(out):
        os.mkdir(out)

    vocab_path = os.path.join(out, 'vocab.json')

    with open(vocab_path, 'w') as f:
        json.dump(vocab, f)
    model_path = os.path.join( out, 'best_model.npz')
    model_setup = args_dict
    model_setup['vocab_path'] = vocab_path
    model_setup['model_path'] = model_path
    model_setup['n_class'] = n_class
    model_setup['datetime'] = current_datetime
    with open(os.path.join(out, 'args.json'), 'w') as f:
        json.dump(args_dict, f)
    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()