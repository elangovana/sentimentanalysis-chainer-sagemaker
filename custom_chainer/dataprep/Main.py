import argparse
import csv

import os

import logging

from dataprep.Splitter import Splitter

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def write_csv(outputfile, dataset):
    with open(outputfile, "w") as handle:
        csv_writer = csv.writer(handle, delimiter=',', quotechar='"')
        for l in dataset:
            csv_writer.writerow(l)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile",
                        help="The input file, yelpreviews.csv to split")

    parser.add_argument("outdir",
                        help="The output directory")

    parser.add_argument("--use-in-memory",
                        help="This will load entire file into memory for ultrfast performance . But you may run into out-of moery error if you dont have sufficient memory..", default="N",  choices={'Y', 'N'})

    parser.add_argument("--first-file-name",
                        help="The output directory", default="train.shuffled.csv")

    parser.add_argument("--second-file-name",
                        help="The output directory", default="test.shuffled.csv")

    parser.add_argument("--divide",
                        help="Whether to divide, Enter Y or N. If you choose Y, then --first-file-name and --second-file-name  & --use-in-memory are ignored ",
                        default="N", choices={'Y', 'N'})
    args = parser.parse_args()
    print(args.__dict__)


    logger.info("Starting ... ")
    use_in_mem_cache  = args.use_in_memory == "Y"

    if args.divide == "Y":
        splitter = Splitter(args.inputfile, args.outdir).split(args.outdir)
    else:
        first, second = Splitter(args.inputfile, args.outdir).split_traintest(use_in_memory_shuffle=use_in_mem_cache)
        write_csv(os.path.join(args.outdir, "train.shuffled.csv"), first)
        write_csv(os.path.join(args.outdir, "test.shuffled.csv"), second)

    logger.info("Completed ... ")
