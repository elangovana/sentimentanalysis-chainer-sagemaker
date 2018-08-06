import argparse
import csv

import chainer
import os

import dataprep.YelpChainerDataset


def split(file, first_size_percent=.7, seed=1572):
    dataset = dataprep.YelpChainerDataset.YelpChainerDataset(file, delimiter=",", encoding="utf-8", quote_charcter='"')
    print(len(dataset))
    first_size= int( len(dataset)*first_size_percent)
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

    args = parser.parse_args()
    first, second = split(args.inputfile)
    write_csv(os.path.join(args.outdir, "train.csv"), first)
    write_csv(os.path.join(args.outdir, "test.csv"), second)