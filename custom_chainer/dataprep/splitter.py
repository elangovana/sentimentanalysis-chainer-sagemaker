import argparse
import csv

import chainer
import os

import dataprep.YelpChainerDataset
import logging

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger(__name__)


def split_traintest(file_or_dir, first_size_fraction=.8, seed=1572) -> tuple(
    (chainer.datasets.sub_dataset, chainer.datasets.sub_dataset)):
    """
Splits a file into 2  sets such as training & test.
    :param file_or_dir: The file to split
    :param first_size_fraction: The fraction of data to be polaced in the first set. Say if this value is .7, then 70% os the data is placed in the first set
    :param seed: The random seed to fix
    :return: 2 datasets
    """
    if os.path.isfile(file_or_dir):
        dataset = dataprep.YelpChainerDataset.YelpChainerDataset(file_or_dir, delimiter=",", encoding="utf-8",
                                                                 quote_charcter='"')
    else:
        dataset = chainer.datasets.ConcatenatedDataset(*get_dataset_array(file_or_dir))
    first_size = int(len(dataset) * first_size_fraction)
    first, second = chainer.datasets.split_dataset_random(dataset, first_size, seed=seed)
    return first, second


def get_dataset_array(base_dir):
    # Can make this dynamic, but in this sample 2 parts of the file
    datasets = []
    for f in os.listdir(base_dir):
        full_path = os.path.join(base_dir, f)
        datasets.append(
            dataprep.YelpChainerDataset.YelpChainerDataset(full_path, delimiter=",", encoding="utf-8",
                                                           quote_charcter='"'))

    return datasets


def split(filepath, outputdir, has_header=True, no_of_parts=None, delimiter=",", encoding="utf-8", quote_character='"'):
    # Approximate each line is 2KB
    KB = 1024
    approx_size_of_each_line = 1 * KB
    approx_total_lines = os.path.getsize(filepath) / approx_size_of_each_line

    # Get number of lines per part
    MB = 2 * (KB * KB)
    no_of_parts = no_of_parts or int(os.path.getsize(filepath) / MB) + 1
    no_lines_per_part = int(approx_total_lines / no_of_parts)

    logger.info("Dividing file {} into estimated {} parts".format(filepath, no_of_parts))

    with open(filepath, encoding=encoding) as handle:
        csv_reader = csv.reader(handle, delimiter=delimiter, quotechar=quote_character)
        # Skip first line if header
        header = None
        if has_header:
            header = next(csv_reader)

        # Count the number of lines
        part_index = 0
        end_of_file = False

        while (not end_of_file):
            part_index = part_index + 1
            part_name = "{}_part_{:03d}.csv".format(os.path.basename(filepath), part_index)
            output_part = os.path.join(outputdir, part_name)
            end_of_file = write_part_to_file(csv_reader, output_part, delimiter, quote_character, header, no_lines_per_part)

    logger.info("Completed dividing files sucessfully")


def write_part_to_file(input_csv_reader, output_file_name, delimiter, quote_character, header, max_no_lines):
    end_of_file = False
    with open(output_file_name, "w") as handle:
        csv_writer = csv.writer(handle, delimiter=delimiter, quotechar=quote_character)
        if header is not None:
            csv_writer.writerow(header)

        for i in range(0, max_no_lines):
            try:
                line = next(input_csv_reader)
                csv_writer.writerow(line)
            except StopIteration:
                end_of_file = True
                break
        logger.info("Completed part {}".format(output_file_name))
    return end_of_file


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

    parser.add_argument("--first-file-name",
                        help="The output directory", default="train.shuffled.csv")
    parser.add_argument("--second-file-name",
                        help="The output directory", default="test.shuffled.csv")

    parser.add_argument("--divide",
                        help="Whether to divide, Enter Y or N. If you choose Y, then --first-file-name and --second-file-name are ignored ",
                        default="N", choices={'Y', 'N'})
    args = parser.parse_args()
    print(args.__dict__)

    logger.info("Starting ... ")
    if args.divide == "Y":
        split(args.inputfile, args.outdir)
    else:
        first, second = split_traintest(args.inputfile)
        write_csv(os.path.join(args.outdir, "train.shuffled.csv"), first)
        write_csv(os.path.join(args.outdir, "test.shuffled.csv"), second)

    logger.info("Completed ... ")
