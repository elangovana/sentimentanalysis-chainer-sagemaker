import csv
import logging
import os

import chainer

import dataprep.YelpChainerDataset


class Splitter:
    def __init__(self, file_or_dir, has_header=True, delimiter=",", encoding="utf-8", quote_charcter='"'):
        self.has_header = has_header
        self.quote_character = quote_charcter
        self.encoding = encoding
        self.delimiter = delimiter
        self.file_or_dir = file_or_dir
        self._logger = logging.getLogger(__name__)

    def split_traintest(self, first_size_fraction=.8, seed=1572, use_in_memory_shuffle=False) -> tuple(
        (chainer.datasets.sub_dataset, chainer.datasets.sub_dataset)):
        """
    Splits a file into 2  sets such as training & test.
        :param file_or_dir: The file to split
        :param first_size_fraction: The fraction of data to be polaced in the first set. Say if this value is .7, then 70% os the data is placed in the first set
        :param seed: The random seed to fix
        :return: 2 datasets
        """

        # Prepare dataset
        if os.path.isfile(self.file_or_dir):
            dataset = dataprep.YelpChainerDataset.YelpChainerDataset(self.file_or_dir, delimiter=self.delimiter,
                                                                     encoding=self.encoding,
                                                                     quote_charcter=self.quote_character,
                                                                     has_header=self.has_header,
                                                                     use_in_memory_shuffle=use_in_memory_shuffle)
        else:
            dataset = chainer.datasets.ConcatenatedDataset(
                *self._get_dataset_array(self.file_or_dir, use_in_memory_shuffle=use_in_memory_shuffle))

        # Split
        first_size = int(len(dataset) * first_size_fraction)
        first, second = chainer.datasets.split_dataset_random(dataset, first_size, seed=seed)
        return first, second

    def _get_dataset_array(self, base_dir, use_in_memory_shuffle):
        # Can make this dynamic, but in this sample 2 parts of the file
        datasets = []
        for f in os.listdir(base_dir):
            full_path = os.path.join(base_dir, f)
            datasets.append(
                dataprep.YelpChainerDataset.YelpChainerDataset(full_path, delimiter=self.delimiter,
                                                               encoding=self.encoding,
                                                               quote_charcter=self.quote_character,
                                                               use_in_memory_shuffle=use_in_memory_shuffle))

        return datasets

    def split(self, outputdir, no_of_parts=None):
        if not os.path.isfile(self.file_or_dir):
            raise ValueError("The constructor argument file_or_dir {} must be a file to invoke this operation.".format(
                self.file_or_dir))
        # Approximate each line is 2KB
        KB = 1024
        approx_size_of_each_line = 1 * KB
        approx_total_lines = os.path.getsize(self.file_or_dir) / approx_size_of_each_line

        # Get number of lines per part
        MB = 2 * (KB * KB)
        no_of_parts = no_of_parts or int(os.path.getsize(self.file_or_dir) / MB) + 1
        no_lines_per_part = int(approx_total_lines / no_of_parts)

        self._logger.info("Dividing file {} into estimated {} parts".format(self.file_or_dir, no_of_parts))

        with open(self.file_or_dir, encoding=self.encoding) as handle:
            csv_reader = csv.reader(handle, delimiter=self.delimiter, quotechar=self.quote_character)
            # Skip first line if header
            header = None
            if self.has_header:
                header = next(csv_reader)

            # Count the number of lines
            part_index = 0
            end_of_file = False

            while (not end_of_file):
                part_index = part_index + 1
                part_name = "{}_part_{:03d}.csv".format(os.path.basename(self.file_or_dir), part_index)
                output_part = os.path.join(outputdir, part_name)
                end_of_file = self._write_part_to_file(csv_reader, output_part, no_lines_per_part, header)

        self._logger.info("Completed dividing files sucessfully")

    def _write_part_to_file(self, input_csv_reader, output_file_name, max_no_lines, header):
        end_of_file = False
        with open(output_file_name, "w") as handle:
            csv_writer = csv.writer(handle, delimiter=self.delimiter, quotechar=self.quote_character)
            if header is not None:
                csv_writer.writerow(header)

            for i in range(0, max_no_lines):
                try:
                    line = next(input_csv_reader)
                    csv_writer.writerow(line)
                except StopIteration:
                    end_of_file = True
                    break
            self._logger.info("Completed part {}".format(output_file_name))
        return end_of_file
