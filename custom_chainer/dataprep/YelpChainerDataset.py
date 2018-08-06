import csv
import io

import chainer


class YelpChainerDataset(chainer.dataset.iterator.Iterator):

    def __init__(self, file, has_header=False, delimiter=",", quote_charcter='"', encoding='utf-8'):

        self.quote_charcter = quote_charcter
        self.delimiter = delimiter
        self.encoding = encoding
        self.has_header = has_header
        self.length = None
        self.filepath = file

    def __getitem__(self, idx):
        # TODO: Try and use linecache to optimise performance. The problem is that this is a multiline file
        with io.open(self.filepath, encoding=self.encoding) as handle:
            csv_reader = csv.reader(handle, delimiter=self.delimiter, quotechar=self.quote_charcter)

            # Skip first line if header
            if self.has_header: next(csv_reader)

            # Loop through until we find the record
            i = 0
            line = ""
            while i <= idx:
                line = next(csv_reader)
                i = i + 1

            return line

    def __len__(self):
        if self.length is None:
            self.length = self.getcount()
        return self.length

    def getcount(self):
        total_records = 0
        # Get file length
        with io.open(self.filepath, encoding=self.encoding) as handle:
            csv_reader = csv.reader(handle, delimiter=',', quotechar='"')
            # Skip first line if header
            if self.has_header: next(csv_reader)

            # Count the number of lines
            for l in csv_reader:
                total_records = total_records + 1
        return total_records
