import multiprocessing
from multiprocessing import Pool
import csv
import nltk
import threading
import logging


class BlazingTextFormatter:
    def __init__(self):
        # TODO: Move this to set up, otherwise hard to unit test
        nltk.download('punkt')

    @property
    def logger(self):
        return logging.getLogger(__name__)

    def format(self, iterator, label_index, text_index, outputhandle, max_process=multiprocessing.cpu_count() - 1):
        # Initialise queues, processor pool
        produce_pool = Pool(processes=max_process)
        m = multiprocessing.Manager()
        q = m.Queue()
        csv_writer = csv.writer(outputhandle, delimiter=' ', lineterminator='\n')

        # Multiprocess format
        self.logger.info("Running process pool to parse text")
        [produce_pool.apply(self._producer, args=(x[label_index], x[text_index], q,)) for x in iterator]
        self.logger.info("Completed process pool")

        # Start the consumer thread
        consumer = threading.Thread(target=self._consumer, args=(q, csv_writer))
        consumer.start()

        # Close processes and thread
        produce_pool.close()
        produce_pool.join()
        q.put(None)
        consumer.join()

    def _producer(self, label, text, q):
        line = self.format_line(text, label)
        q.put(line)
        return line

    def _consumer(self, q, csv_writer):
        i = 0
        while True:
            item = q.get()
            # if item poison pill quit
            if item is None:
                break
            i += 1
            if i%10000 == 0: self.logger.info("Formatted {} lines so far".format(i))
            # write item to file
            self._write_line(csv_writer, item)

    def format_line(self, text, label):
        cur_row = []
        label = "__label__" + label  # Prefix the index-ed label with __label__
        cur_row.append(label)
        cur_row.extend(nltk.word_tokenize(text.lower()))
        return cur_row

    def _write_line(self, csv_writer, record):
        csv_writer.writerow(record)
