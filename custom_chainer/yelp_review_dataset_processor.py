import csv
import io

import numpy

from custom_chainer.nlp_utils import make_vocab, transform_to_array, split_text, normalize_text


class YelpReviewDatasetProcessor:
    def get_dataset(self, name_path_list, vocab=None, shrink=1,
                    char_based=False, seed=777):
        datasets = name_path_list
        train = self.read_data(
            datasets[0], char_based=char_based)
        if len(datasets) == 2:
            test = self.read_data(
                datasets[1], char_based=char_based)
        else:
            numpy.random.seed(seed)
            alldata = numpy.random.permutation(train)
            train = alldata[:-len(alldata) // 10]
            test = alldata[-len(alldata) // 10:]

        if vocab is None:
            print('constract vocabulary based on frequency')
            vocab = make_vocab(train)

        train = transform_to_array(train, vocab)
        test = transform_to_array(test, vocab)

        return train, test, vocab


    def read_data(self, path, char_based=False):
        """
        Expect a file with header in the following format
        "review_id","user_id","business_id","stars","date","text","useful","funny","cool"
        :param path:

        :param char_based:
        :return:
        """

        with io.open(path, encoding='utf-8', errors='ignore') as f:
            dataset = self.parse_csv(f, char_based)
        return dataset

    def parse_csv(self, handle, char_based, has_header=True):
        csv_reader = csv.reader(handle, delimiter=',', quotechar='"')
        # Ignore header
        if has_header: next(csv_reader)
        dataset = []
        for l in csv_reader:
            tokens, label = self.process_line(char_based, l)
            dataset.append((tokens, label))
        return  dataset

    def process_line(self, char_based,  l):
        # Create label

        rating = int(l[3])
        label = 0
        if rating > 3:
            label = 1
        elif rating < 3:
            label = -1
        review_text = l[5]
        tokens = self.extract_tokens(char_based, review_text)
        return tokens,label


    def extract_tokens(self, char_based, review_text):
        tokens = split_text(normalize_text(review_text), char_based)
        return tokens
