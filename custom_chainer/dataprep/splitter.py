import argparse
import csv

import chainer
import os

import dataprep.YelpChainerDataset


def split(file, first_size_fraction=.8, seed=1572) -> tuple((chainer.datasets.sub_dataset, chainer.datasets.sub_dataset)):
    """
Splits a file into 2  sets such as training & test.
    :param file: The file to split
    :param first_size_fraction: The fraction of data to be polaced in the first set. Say if this value is .7, then 70% os the data is placed in the first set
    :param seed: The random seed to fix
    :return: 2 datasets
    """
    dataset = dataprep.YelpChainerDataset.YelpChainerDataset(file, delimiter=",", encoding="utf-8", quote_charcter='"')
    first_size= int(len(dataset) * first_size_fraction)
    first, second = chainer.datasets.split_dataset_random(dataset, first_size, seed=seed)
    return first,second


def write_csv(outputfile, dataset):
    with open(outputfile, "w") as handle:
        csv_writer = csv.writer(handle, delimiter=',', quotechar='"')
        for l in dataset:
            csv_writer.writerow(l)



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputfile",
                        help="The input file, yelpreviews.csv to split")

    parser.add_argument("outdir",
                        help="The output directory")

    parser.add_argument("--first-file-name",
                        help="The output directory", default= "train.shuffled.csv")
    parser.add_argument("--second-file-name",
                        help="The output directory", default= "test.shuffled.csv")
    args = parser.parse_args()
    print(args.__dict__)
    first, second = split(args.inputfile)
    write_csv(os.path.join(args.outdir, "train.shuffled.csv"), first)
    write_csv(os.path.join(args.outdir, "test.shuffled.csv"), second)