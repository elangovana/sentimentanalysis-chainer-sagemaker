import tempfile
from logging.config import fileConfig
from unittest import TestCase

import os
from ddt import ddt, data, unpack



from TrainPipelineBuilder import TrainPipelineBuilder
from dataprep.YelpChainerDataset import YelpChainerDataset
from dataprep.YelpReviewDatasetProcessor import YelpReviewDatasetProcessor
from main_full_cycle import model_fn, input_fn, predict_fn, output_fn

from pathlib import Path

from moviesdataset.MovieDatasetIterator import MovieDatasetIterator
from moviesdataset.MovieDatasetIteratorProcessor import MovieDatasetIteratorProcessor


@ddt
class TestModel_fn(TestCase):
    def setUp(self):
        fileConfig(os.path.join(os.path.dirname(__file__), 'logger.ini'))


    @data(("data/sample_train.csv", "data", "data/sample_test.csv", 300, None)
        , ("data/sample_train.csv", "data", "data/sample_test.csv", 3, "data/sample_embed_3d.txt")
          )
    @unpack
    def test_model_fn(self, dataset, out_dir, testdata, unit, embed_file):
        # Arrange

        full_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset)
        full_out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), out_dir)
        full_embed = None
        if embed_file is not None:
            full_embed = os.path.join(os.path.dirname(os.path.realpath(__file__)), embed_file)

        with tempfile.TemporaryDirectory(suffix=None, prefix=None, dir=full_out_dir) as temp_dir:
            temp_out = temp_dir
            full_testset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), testdata)
            test_data_string = Path(full_testset_path).read_text()
            # Run Train


            builder = TrainPipelineBuilder(data_has_header=True, batchsize=10, char_based=False,
                                           dropout=.5,
                                           epoch=10, gpus=None, no_layers=1,
                                           embed_dim=unit, embedding_file=full_embed, max_vocab_size=20000,
                                           min_word_frequency=1)

            iterator = YelpChainerDataset(full_dataset_path, has_header=True)
            data_processor = YelpReviewDatasetProcessor(iterator)
            builder.run(data_processor, 3, "cnn", temp_out)
            # Act + Assert
            model = model_fn(temp_out)

            # TODO Clean up, pass this as input
            input_object = input_fn(test_data_string.encode('utf-8'), "text/plain")
            prediction = predict_fn(input_object, model)
            actual = output_fn(prediction, "text/plain")

        # assert
        print(actual)
        self.assertIsNotNone(actual)

    @data(("data/movies_sample_dataset.csv", "data", "data/sample_test.csv", 300, None)
        , ("data/movies_sample_dataset.csv", "data", "data/sample_test.csv", 3, "data/sample_embed_3d.txt")
          )
    @unpack
    def test_model_movies_fn(self, dataset, out_dir, testdata, unit, embed_file):
        # Arrange

        full_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), dataset)
        full_out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), out_dir)
        full_embed = None
        if embed_file is not None:
            full_embed = os.path.join(os.path.dirname(os.path.realpath(__file__)), embed_file)

        with tempfile.TemporaryDirectory(suffix=None, prefix=None, dir=full_out_dir) as temp_dir:
            temp_out = temp_dir
            full_testset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), testdata)
            test_data_string = Path(full_testset_path).read_text()
            # Run Train

            builder = TrainPipelineBuilder( data_has_header=True, batchsize=10, char_based=False,
                                           dropout=.5,
                                           epoch=10, gpus=None, no_layers=1,
                                           embed_dim=unit, embedding_file=full_embed, max_vocab_size=20000,
                                           min_word_frequency=1)
            iterator = MovieDatasetIterator(full_dataset_path)
            data_processor = MovieDatasetIteratorProcessor(iterator)
            builder.run(data_processor, 3, "cnn", temp_out)
            # Act + Assert
            model = model_fn(temp_out)

            # # TODO Clean up, pass this as input
            input_object = input_fn(test_data_string.encode('utf-8'), "text/plain")
            prediction = predict_fn(input_object, model)
            actual = output_fn(prediction, "text/plain")

        #assert
        print(actual)
        self.assertIsNotNone(actual)
