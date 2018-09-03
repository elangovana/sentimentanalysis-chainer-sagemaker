import argparse

from predict import run_batch,  run_online


def run_test():
    #global model, vocab, setup
    model, vocab, setup = Train(args)
    if args.gpu >= 0:
        run_batch(args.gpu,model,vocab,setup)
    else:
        run_online(args.gpu, args.testset,model, vocab, setup)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Chainer example: Text Classification')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model-setup', required=True,
                        help='Model setup dictionary.')

    parser.add_argument('--testset', required=True,
                        help='Model setup dictionary.')
    args = parser.parse_args()

    run_test()