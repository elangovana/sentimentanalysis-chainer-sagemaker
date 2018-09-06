import argparse
import logging

import multiprocessing

from chainer import iterators

from dataformatter.blazingTextFormatter import BlazingTextFormatter
from datasetyelp.YelpChainerDataset import YelpChainerDataset
from datasetyelp.YelpReviewDatasetProcessor import YelpReviewDatasetProcessor



class YelpBlazingTextFormatter:
    def __init__(self, formatter=None, label_extractor=None, text_extractor=None):
        self.label_extractor = label_extractor or YelpReviewDatasetProcessor.get_label
        self.text_extractor = text_extractor or YelpReviewDatasetProcessor.get_review_text
        self.formatter = formatter or BlazingTextFormatter()

    def __call__(self, iterator, outputhandle, number_of_process=multiprocessing.cpu_count()):
        label_index, text_index = 0, 1
        self.formatter.format(self.clean(iterator), label_index, text_index, outputhandle,
                              max_process=number_of_process)

    def clean(self, iterator):
        for r in iterator:
            label = self.label_extractor(r)
            text = self.text_extractor(r)
            yield (str(label), text)


if __name__ == '__main__':
    FORMAT = '%(asctime)s %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument("yelpreviewfile",
                        help="Yelp review file")

    parser.add_argument("outfile",
                        help="The output file")

    parser.add_argument("--process-count",
                        help="The number of processes to use for multiprocessing", default=1, type=int)
    args = parser.parse_args()

    logger.info("Starting ... ")
    formatter = YelpBlazingTextFormatter()
    iterator = YelpChainerDataset(args.yelpreviewfile, has_header=True)

    with open(args.outfile, "w") as out:
        formatter(iterator, out, number_of_process=args.process_count)
    logger.info("Completed ... ")
