import argparse
import json
import os
import logging
from predict import get_model, get_formatted_input, predict, \
    get_formatted_output
from train import run_train

#TODO: Check why logging doesnt work on SageMaker
logging.basicConfig(level=logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

def train():
    logger = logging.getLogger(__name__)
    print("Log level is {}".format(logger.level))

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
    parser.add_argument('--gpu', '-g', type=int, default=int(os.environ.get('SM_NUM_GPUS', 0))-1,
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


    logger.info("Invoking training with arguments \n ..",json.dumps(args.__dict__, indent=2))


    #args parse


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

    run_train(batchsize, char_based,  dataset, dropout, epoch, gpu, model, no_layers, out,
              unit)


def model_fn(model_dir):
    model_full_path = model_dir
    # This is a hack because the model is placed in a sub directory within model_dir..
    for root, subFolder, files in os.walk(model_dir):
        for item in files:
            if (item == "args.json"):
                model_full_path = root
                break

    return get_model(model_full_path,os.environ.get('SM_NUM_GPUS', 0) - 1 )



def input_fn(request_body, request_content_type):
    logger = logging.getLogger(__name__)
    logger.debug("Calling input fomatter for content_type ..{}".format(request_content_type))
    return get_formatted_input(request_body, request_content_type)



def predict_fn(input_object, model):

    return predict(input_object, model, os.environ.get('SM_NUM_GPUS', 0)-1)


# Serialize the prediction result into the desired response content type
def output_fn(prediction, response_content_type):
    logger = logging.getLogger(__name__)

    logger.debug("Calling output for content_type ..{}".format(response_content_type))
    print(response_content_type)
    result =    get_formatted_output(prediction, response_content_type)
    logger.debug("Received output {} ..".format(result))
    return result


if __name__ == '__main__':
    train()
