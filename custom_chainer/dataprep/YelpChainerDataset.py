import csv
import io
from io import StringIO

import chainer
import logging


class YelpChainerDataset(chainer.dataset.iterator.Iterator):

    def __init__(self, file, has_header=False, delimiter=",", quote_charcter='"', encoding='utf-8',
                 use_in_memory_shuffle=True):
        self.use_in_memory_shuffle = use_in_memory_shuffle
        self._logger = logging.getLogger(__name__)
        self.quote_charcter = quote_charcter
        self.delimiter = delimiter
        self.encoding = encoding
        self.has_header = has_header
        self._header = None
        self.length = None
        self.filepath = file
        self.line_dict = None

    def __getitem__(self, idx):
        # If use_in_memory, load file contents to memory..
        if self.use_in_memory_shuffle:
            line = self._get_line_from_memory(idx)
        # If optmise memory , but slow based on disk..
        else:
            # TODO: Try and use linecache to optimise performance. The problem is that this is a multiline file
            line = self._get_line_from_disk(idx)

        return line

    def _get_line_from_disk(self, idx):
        with io.open(self.filepath, encoding=self.encoding) as handle:
            line = self._getline(handle, idx)
        return line

    def _get_line_from_memory(self, idx):

        if self.line_dict is None:
            # Load the contents into memory and store as hash
            with io.open(self.filepath, encoding=self.encoding) as handle:
                reader = self._get_reader(handle)
                # Ignore header
                if self.has_header:
                    self._header = next(reader)

                i = 0
                self.line_dict = {}
                for l in reader:
                    self.line_dict[i] = l
                    i = i + 1

        line = self.line_dict[idx]
        return line

    def _getline(self, handle, idx):
        csv_reader = self._get_reader(handle)
        # Skip first line if header
        if self.has_header:
            self._header = next(csv_reader)

        # Loop through until we find the record
        i = 0
        line = ""
        self._logger.debug("Reading line {} from file {}".format(idx, self.filepath))
        while i <= idx:
            line = next(csv_reader)
            i = i + 1
        self._logger.debug("Read complete line {} from file {}".format(idx, self.filepath))
        return line

    def __len__(self):
        if self.length is None:
            self.length = self.getcount()
        return self.length

    def getcount(self):
        total_records = 0
        # Get file length
        with io.open(self.filepath, encoding=self.encoding) as handle:
            csv_reader = self._get_reader(handle)
            # Skip first line if header
            if self.has_header: next(csv_reader)

            # Count the number of lines
            for l in csv_reader:
                total_records = total_records + 1
        return total_records

    def _get_reader(self, handle):
        return csv.reader(handle, delimiter=self.delimiter, quotechar=self.quote_charcter)

