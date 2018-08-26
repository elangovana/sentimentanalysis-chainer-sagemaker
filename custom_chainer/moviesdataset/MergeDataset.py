import csv


class MergeDataset:

    def __call__(self, positive_handle, negative_handle, out_handle, delimiter=",", quote_character='"'):
        csv_writer = csv.writer(out_handle, delimiter=delimiter, quotechar=quote_character)

        # Write positive
        for r in positive_handle:
            csv_writer.writerow([r.strip("\n"), 1])

        # Write negative
        for r in negative_handle:
            csv_writer.writerow([r.strip("\n"), 0])
