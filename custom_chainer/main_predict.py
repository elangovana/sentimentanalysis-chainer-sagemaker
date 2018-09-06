import argparse

from TrainPipelineBuilder import TrainPipelineBuilder
from predict import run_batch, run_online


def run_test(model_dir):
    # global model, vocab, setup
    model, vocab, setup, _ = TrainPipelineBuilder.load(model_dir)
    if args.gpu >= 0:
        run_batch(args.gpu, model, vocab, setup)
    else:
        run_online(args.gpu, args.testset, model, vocab, setup)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Chainer example: Text Classification')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model-dir', required=True,
                        help='The directory containing the vcoab, model and the args.json file')

    parser.add_argument('--testset', required=True,
                        help='Model setup dictionary.')
    args = parser.parse_args()

    run_test(args.model_dir)
