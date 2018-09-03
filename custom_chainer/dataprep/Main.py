import argparse
import logging
import os

from dataprep.Splitter import Splitter

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    shuffle = 'shuffle'
    split = 'split'
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile",
                        help="The input file, yelpreviews.csv to split")

    parser.add_argument("outdir",
                        help="The output directory")

    parser.add_argument("operation",
                        help="Whether to shuffle or split the file. If you choose split, then --first-file-name and --second-file-name  & --use-in-memory are ignored ",
                        choices={shuffle, split})

    parser.add_argument("--num-cpu",
                        help="This option applies only for shuffle op. Will multiprocess by default equal to the number of CPUs",
                        default=None, type=int)

    parser.add_argument("--use-in-memory",
                        help="This will load entire file into memory for ultrfast performance, applies only when --divide N. But you may run into out-of memory error if you dont have sufficient memory..",
                        default="Y", choices={'Y', 'N'})

    parser.add_argument("--first-file-name",
                        help="The output directory", default="train.shuffled.csv")

    parser.add_argument("--second-file-name",
                        help="The output directory", default="test.shuffled.csv")

    args = parser.parse_args()
    print(args.__dict__)

    logger.info("Starting ... ")
    use_in_mem_cache = args.use_in_memory == "Y"

    splitter = Splitter(args.inputfile, args.outdir)
    if args.operation == split:
        splitter = splitter.split(args.outdir)
    else:
        # Shuffle
        with open(os.path.join(args.outdir, "train.shuffled.csv"), "w") as first:
            with open(os.path.join(args.outdir, "test.shuffled.csv"), "w") as second:
                splitter.shuffleandsplit(first, second, use_in_memory_shuffle=use_in_mem_cache,
                                         n_processes=args.num_cpu)

    logger.info("Completed ... ")
