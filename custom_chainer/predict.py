import argparse
import csv
import json
import sys
from io import StringIO

import chainer
import os

from custom_chainer import nlp_utils, nets
from custom_chainer.yelp_review_dataset_processor import YelpReviewDatasetProcessor


def setup_model(args):
    sys.stderr.write(json.dumps(args.__dict__, indent=2) + '\n')
    setup = json.load(open(args.model_setup))
    gpu = args.gpu
    sys.stderr.write(json.dumps(setup, indent=2) + '\n')

    model, vocab = extract_model(gpu, setup, setup['vocab_path'], setup['model_path'])

    return model, vocab, setup


def extract_model(gpu, setup, vocab_path, model_path):
    with open(vocab_path) as vocab_handle:
        vocab = json.load(vocab_handle)
    n_class = setup['n_class']
    # Setup a model
    if setup['model'] == 'rnn':
        Encoder = nets.RNNEncoder
    elif setup['model'] == 'cnn':
        Encoder = nets.CNNEncoder
    elif setup['model'] == 'bow':
        Encoder = nets.BOWMLPEncoder
    encoder = Encoder(n_layers=setup['no_layers'], n_vocab=len(vocab),
                      n_units=setup['unit'], dropout=setup['dropout'])
    model = nets.TextClassifier(encoder, n_class)
    chainer.serializers.load_npz(model_path, model)
    if gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()  # Copy the model to the GPU
    return model, vocab


def run_online(gpu, testdatafile, model, vocab, setup_json):
    # predict labels online
    data_processor = YelpReviewDatasetProcessor()
    test_data = data_processor.read_data(testdatafile, setup_json['char_based'])
    tokens_list = [r[0] for r in test_data]
    run_inference(gpu, tokens_list, model, vocab)


def run_inference(gpu, test_data, model, vocab):
    result = []
    for row in test_data:
        tokens=row
        print(tokens)
        xs = nlp_utils.transform_to_array([tokens], vocab, with_label=False)
        xs = nlp_utils.convert_seq(xs, device=gpu, with_label=False)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            prob = model.predict(xs, softmax=True)[0]
        answer = int(model.xp.argmax(prob))
        score = float(prob[answer])
        # print('{}\t{:.4f}\t{}'.format(answer, score, ' '.join(tokens)))
        print('{}\t{:.4f}'.format(answer, score))
        result.append((answer, score))


def run_batch(gpu, model, vocab, setup_json, batchsize=64):
    # predict labels by batch

    def predict_batch(words_batch):
        xs = nlp_utils.transform_to_array(words_batch, vocab, with_label=False)
        xs = nlp_utils.convert_seq(xs, device=gpu, with_label=False)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            probs = model.predict(xs, softmax=True)
        answers = model.xp.argmax(probs, axis=1)
        scores = probs[model.xp.arange(answers.size), answers].tolist()
        for words, answer, score in zip(words_batch, answers, scores):
            #print('{}\t{:.4f}\t{}'.format(answer, score, ' '.join(words)))
            print('{}\t{:.4f}'.format(answer, score))

    batch = []
    for l in sys.stdin:
        l = l.strip()
        if not l:
            if batch:
                predict_batch(batch)
                batch = []
            print('# blank line')
            continue
        text = nlp_utils.normalize_text(l)
        words = nlp_utils.split_text(text, char_based=setup_json['char_based'])
        batch.append(words)
        if len(batch) >= batchsize:
            predict_batch(batch)
            batch = []
    if batch:
        predict_batch(batch)

def get_model(model_dir, gpu = -1):
    with open(os.path.join(model_dir, "args.json")) as args_file:
        setup_json =json.load(args_file)

    vocab_path =os.path.join(model_dir,  os.path.basename(setup_json["vocab_path"]))
    model_path =os.path.join(model_dir,  os.path.basename(setup_json["model_path"]))

    model, vocab = extract_model(gpu, setup_json, vocab_path, model_path)
    return  (model, vocab , setup_json)


def get_formatted_input(request_body, request_content_type):

    if request_content_type == "text/plain":

        return parse_csv(StringIO(request_body))
    else:
        raise ValueError("Content_type {} is not recognised".format(request_content_type))

def parse_csv(handle):
    data_processor =YelpReviewDatasetProcessor()
    csv_reader = csv.reader(handle, delimiter=',', quotechar='"')

    dataset = []
    for l in csv_reader:
        tokens = data_processor.extract_tokens(False,l[0])
        dataset.append(tokens)
    return  dataset

def predict(input_object, model, gpu=-1):
    peristsed_model, vocab, setup = model
    return run_inference(gpu, input_object, peristsed_model, vocab)


# Serialize the prediction result into the desired response content type
def get_formatted_output(prediction, response_content_type):
    if response_content_type == "text/plain":
     return json.dumps( prediction)
