import argparse
import json
import logging
import os
import sys

from TrainPipelineBuilder import TrainPipelineBuilder
from datasetmovies.MovieDatasetFactory import MovieDatasetFactory
from datasetyelp.YelpDatasetFactory import YelpDatasetFactory
from predict import get_formatted_input, predict, \
    get_formatted_output


def train():
    logger = logging.getLogger(__name__)

    datasets = {MovieDatasetFactory().name: MovieDatasetFactory(), YelpDatasetFactory().name: YelpDatasetFactory()}
    n_classes = {MovieDatasetFactory().name: 2, YelpDatasetFactory().name: 3}

    parser = argparse.ArgumentParser(
        description='Chainer example: Text Classification')
    parser.add_argument('--dataset-type', required=True,
                        help='The type of dataset', choices=datasets.keys())

    parser.add_argument('--traindata', required=True,
                        help='The dataset file with respect to the train directory')

    parser.add_argument('--traindata-dir',
                        help='The directory containing training artifacts such as training data',
                        default=os.environ.get('SM_CHANNEL_TRAIN', "."))
    parser.add_argument('--testdata', '-td',
                        help='The test dataset file with respect to the test directory', default=None)
    parser.add_argument('--testdata-dir', '-tdir',
                        help='The directory containing test artifacts such as test data',
                        default=os.environ.get('SM_CHANNEL_TEST', "."))
    parser.add_argument('--model-dir',
                        help='The directory containing model, if you want to reload existing weights',
                        default=os.environ.get('SM_CHANNEL_MODEL', None))

    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of records in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=30,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=int(os.environ.get('SM_NUM_GPUS', 0)) - 1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default=os.environ.get('SM_OUTPUT_DATA_DIR', "result_data"),
                        help='Directory to output the result')
    parser.add_argument('--unit', '-u', type=int, default=300,
                        help='Number of units')
    parser.add_argument('--layer', '-l', type=int, default=1,
                        help='Number of layers of RNN or MLP following CNN')
    parser.add_argument('--dropout', '-d', type=float, default=0.5,
                        help='Dropout rate')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--model', '-model', default='cnn',
                        choices=['cnn', 'rnn', 'bow'],
                        help='Name of encoder model type.')
    parser.add_argument('--char-based', action='store_true')
    parser.add_argument('--pretrained-embed-dir', default=os.environ.get('SM_CHANNEL_PRETRAINEDEMBED', None),
                        help='The path of the file containing the pretrained embeddings')
    parser.add_argument('--pretrained-embed', default=None,
                        help='The name of the file containing the pretrained embeddings within the --pretrained-embed-dir directory')
    args = parser.parse_args()

    logger.info("Invoking training with arguments \n ..{}".format(json.dumps(args.__dict__, indent=2)))

    # args parse

    batchsize = args.batchsize
    encoder_name = args.model
    dataset_type = args.dataset_type
    training_set = datasets[dataset_type].get_dataset(os.path.join(args.traindata_dir, args.traindata))
    validation_set = datasets[dataset_type].get_dataset(
        os.path.join(args.testdata_dir, args.testdata)) if args.testdata is not None else None
    char_based = args.char_based
    no_layers = args.layer
    unit = args.unit
    dropout = args.dropout
    gpus = [g for g in range(0, args.gpu)]
    epoch = args.epoch
    out = args.out
    n_class = n_classes[dataset_type]
    learning_rate = args.learning_rate

    pretrained_embeddings = None
    if (args.pretrained_embed_dir is not None and args.pretrained_embed is not None):
        pretrained_embeddings = os.path.join(args.pretrained_embed_dir, args.pretrained_embed)

    model_dir = args.model_dir

    builder = TrainPipelineBuilder(data_has_header=False, batchsize=batchsize, char_based=char_based, dropout=dropout,
                                   epoch=epoch, gpus=gpus, no_layers=no_layers,
                                   embed_dim=unit, embedding_file=pretrained_embeddings, max_vocab_size=20000,
                                   min_word_frequency=5, learning_rate=learning_rate)

    builder.run(training_set, n_class, encoder_name, out, validationset_iterator=validation_set, model_dir=model_dir)


def model_fn(model_dir):
    model_full_path = model_dir
    # This is a hack because the model is placed in a sub directory within model_dir..
    for root, subFolder, files in os.walk(model_dir):
        for item in files:
            if (item == "args.json"):
                model_full_path = root
                break

    model, vocab, setup_json, _ = TrainPipelineBuilder.load(model_full_path)
    return (model, vocab, setup_json)


def get_gpus():
    no_of_gpus = os.environ.get('SM_NUM_GPUS', 0)
    return [g for g in range(0, no_of_gpus)]


def input_fn(request_body, request_content_type):
    logger = logging.getLogger(__name__)
    logger.debug("Calling input fomatter for content_type ..{}".format(request_content_type))
    return get_formatted_input(request_body, request_content_type)


def predict_fn(input_object, model):
    return predict(input_object, model, get_gpus())


# Serialize the prediction result into the desired response content type
def output_fn(prediction, response_content_type):
    logger = logging.getLogger(__name__)
    logger.debug("Calling output for content_type ..{}".format(response_content_type))
    result = get_formatted_output(prediction, response_content_type)
    logger.debug("Received output {} ..".format(result))
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)],
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    train()
