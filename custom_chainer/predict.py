import csv
import json
import sys
from io import StringIO

import chainer

import gpu_utils
from utils.NlpUtils import normalize_text, split_text, transform_to_array


# #TODO: Cleanup code
# def setup_model(args):
#     sys.stderr.write(json.dumps(args.__dict__, indent=2) + '\n')
#     setup = json.load(open(args.model_setup))
#     gpu = args.gpu
#     sys.stderr.write(json.dumps(setup, indent=2) + '\n')
#
#     model, vocab = extract_model(gpu, setup, setup['vocab_path'], setup['model_path'])
#
#     return model, vocab, setup


# def extract_model(gpu, setup, vocab_path, model_path):
#     with open(vocab_path) as vocab_handle:
#         vocab = json.load(vocab_handle)
#     n_class = setup['n_class']
#     # Setup a model
#     if setup['model'] == 'rnn':
#         Encoder = RNNEncoder.RNNEncoder
#     elif setup['model'] == 'cnn':
#         Encoder = encoders.CNNEncoder.CNNEncoder
#     elif setup['model'] == 'bow':
#         Encoder = encoders.BOWNLPEncoder.BOWMLPEncoder
#     encoder = Encoder(n_layers=setup['no_layers'], n_vocab=len(vocab),
#                       n_units=setup['unit'], dropout=setup['dropout'])
#     model = TextClassifier.TextClassifier(encoder, n_class)
#     chainer.serializers.load_npz(model_path, model)
#     if gpu >= 0:
#         # Make a specified GPU current
#         chainer.backends.cuda.get_device_from_id(gpu).use()
#         model.to_gpu()  # Copy the model to the GPU
#     return model, vocab


def run_online(gpu, testdatafile, model, vocab, setup_json):
    # predict labels online
    with open(testdatafile, "r") as handle:
        test_data = parse_csv(handle, False)
    tokens_list = [r[0] for r in test_data]
    run_inference(gpu, tokens_list, model, vocab)


def run_inference(gpus, test_data, model, vocab):
    result = []
    for row in test_data:
        tokens = row
        print(tokens)
        xs = transform_to_array([tokens], vocab, with_label=False)
        # TODO investigate multi-gpu case
        device = -1
        if len(gpus) > 0:
            device = gpus[0]

        xs = gpu_utils.convert_seq(xs, device=device, with_label=False)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            prob = model.predict(xs, softmax=True)[0]
        answer = int(model.xp.argmax(prob))
        score = float(prob[answer])
        result.append((answer, score))
    return result


def run_batch(gpu, model, vocab, setup_json, batchsize=64):
    # predict labels by batch

    def predict_batch(words_batch):
        xs = transform_to_array(words_batch, vocab, with_label=False)
        xs = gpu_utils.convert_seq(xs, device=gpu, with_label=False)
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            probs = model.predict(xs, softmax=True)
        answers = model.xp.argmax(probs, axis=1)
        scores = probs[model.xp.arange(answers.size), answers].tolist()
        for words, answer, score in zip(words_batch, answers, scores):
            # print('{}\t{:.4f}\t{}'.format(answer, score, ' '.join(words)))
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
        text = normalize_text(l)
        words = split_text(text, char_based=setup_json['char_based'])
        batch.append(words)
        if len(batch) >= batchsize:
            predict_batch(batch)
            batch = []
    if batch:
        predict_batch(batch)


def get_formatted_input(request_body, request_content_type):
    if request_content_type == "text/plain":

        return parse_csv(StringIO(request_body.decode("utf-8")), False)
    else:
        raise ValueError("Content_type {} is not recognised".format(request_content_type))


def parse_csv(handle, char_based):
    csv_reader = csv.reader(handle, delimiter=',', quotechar='"')

    dataset = []
    for l in csv_reader:
        tokens = split_text(normalize_text((l[0])), False)
        dataset.append(tokens)
    return dataset


def predict(input_object, model, gpus=None):
    gpus = gpus or []
    persisted_model, vocab, setup = model
    return run_inference(gpus, input_object, persisted_model, vocab)


# Serialize the prediction result into the desired response content type
def get_formatted_output(prediction, response_content_type):
    if response_content_type == "text/plain" or response_content_type == "*/*":
        return json.dumps(prediction)
