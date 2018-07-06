import tempfile
from unittest import TestCase

import os
from ddt import ddt, data, unpack

from train import model_fn, input_fn, predict_fn, output_fn, run_train

from pathlib import Path

@ddt
class TestModel_fn(TestCase):

    @data(("data/sample_train.csv", "data/result", "data/sample_test.csv"))
    @unpack
    def test_model_fn(self, dataset, out_dir, testdata):
        # Arrange

        full_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset)
        full_out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), out_dir)
        with tempfile.TemporaryDirectory(suffix=None, prefix=None, dir=full_out_dir) as temp_dir:
            temp_out = temp_dir
            full_testset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), testdata)
            test_data_string =  Path(full_testset_path).read_text()
            # Run Train
            run_train(batchsize=10, char_based=False, dataset=[full_dataset_path], dropout=.4, epoch=10, gpu=-1, model="cnn",
                      no_layers=1, out=temp_out, unit=300)

            # Act + Assert
            model = model_fn(temp_out)

            #TODO Clean up, pass this as input
            input_object = input_fn(test_data_string, "text/plain")
            prediction = predict_fn(input_object, model)
            actual = output_fn(prediction, "text/plain")

        #assert
        print(actual)
        self.assertIsNotNone(actual)

