import multiprocessing
from multiprocessing import Pool
import csv
import nltk
import threading

class BlazingTextFormatter:
    def __init__(self):
        nltk.download('punkt')
        # self.queue = multiprocessing.Queue()

    def format(self, iterator, label_index, text_index, outputhandle, max_process=multiprocessing.cpu_count() - 1):
        produce_pool = Pool(processes=max_process)
        m = multiprocessing.Manager()
        q = m.Queue()
        csv_writer = csv.writer(outputhandle, delimiter=' ', lineterminator='\n')
        # Transform
        transformed_rows = [produce_pool.apply(self._producer, args=(x[label_index], x[text_index], q,)) for x in iterator]
        consumer = threading.Thread(target=self._consumer, args=(q, csv_writer))
        consumer.start()
        produce_pool.close()
        produce_pool.join()
        q.put(None)

        consumer.join()
        return transformed_rows

    def _producer(self, label, text, q):
        line = self.format_line(text, label)
        q.put(line)
        return line

    def _consumer(self, q, csv_writer):
        while True:
            item = q.get()
            if item is None:
                break

            self._write_line(csv_writer, item)

    def format_line(self, text, label):
        cur_row = []
        label = "__label__" + label  # Prefix the index-ed label with __label__
        cur_row.append(label)
        cur_row.extend(nltk.word_tokenize(text.lower()))
        return cur_row

    def _write_line(self, csv_writer, record):
        csv_writer.writerow(record)
