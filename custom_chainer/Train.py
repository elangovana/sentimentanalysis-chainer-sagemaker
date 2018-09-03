import datetime
import json
import logging
import os
import chainer
import numpy as np
from chainer import training
from chainer.training import extensions

import TextClassifier
from gpu_utils import convert_seq


# TODO: Cleanup code
class Train:
    def __init__(self,
                 data_processor, encoder, vocab, out_dir, epoch, batchsize, gpus=None):

        self.batchsize = batchsize
        self.gpus = gpus or []
        self.epoch = epoch
        self.out_dir = out_dir

        self.encoder = encoder

        self.vocab = vocab



    @property
    def logger(self):
        return logging.getLogger(__name__)


    def __call__(self, train, test, classifier, snapshot_model_name):
        # Has to be the first line so that the args can be persisted



        # Log some useful info
        self.logger.info(
            '# train data: {}, # test data {}, # vocab size '.format(len(train), len(test), len(self.vocab)))
        n_classes = [int(d[1]) for d in train]
        unique_classes, counts_classes = np.unique(n_classes, return_counts=True)
        self.logger.info("Class distribution: \n{}".format(np.asarray((unique_classes, counts_classes))))

        n_class = len(unique_classes)
        train_iter = chainer.iterators.SerialIterator(train, self.batchsize, shuffle=True)
        test_iter = chainer.iterators.SerialIterator(test, self.batchsize, repeat=False, shuffle=False)

        # Check if can distribute training
        use_distibuted_gpu = False
        use_gpu = False
        main_device = -1

        if len(self.gpus) > 0:
            main_device = 0
            use_gpu = True
            use_distibuted_gpu = len(self.gpus) > 1

        if use_gpu and not use_distibuted_gpu:
            chainer.backends.cuda.get_device_from_id(main_device).use()
            classifier.to_gpu()  # Copy the model to the GPU

        # Setup an optimizer
        optimizer = chainer.optimizers.Adam()
        optimizer.setup(classifier)
        optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))

        # Set up a trainer
        if use_distibuted_gpu:
            device_dict = {}
            for device_id in self.gpus:
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

        trainer = training.Trainer(updater, (self.epoch, 'epoch'), out=self.out_dir)
        # Evaluate the model with the test dataset for each epoch
        trainer.extend(extensions.Evaluator(
            test_iter, classifier,
            converter=convert_seq, device=main_device))
        # Take a best snapshot
        record_trigger = training.triggers.MaxValueTrigger(
            'validation/main/accuracy', (1, 'epoch'))
        trainer.extend(extensions.snapshot_object(
            classifier, snapshot_model_name ),
            trigger=record_trigger)
        # Write a log of evaluation statistics for each epoch
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        # Print a progress bar to stdout
        # trainer.extend(extensions.ProgressBar())
        # Save vocabulary and model's setting

        # Run the training
        trainer.run()




