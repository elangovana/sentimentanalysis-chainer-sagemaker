import argparse
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("positivefile",
                        help="The positive file to merge")

    parser.add_argument("negativefile",
                        help="The negativefile file to merge")

    parser.add_argument("outfile",
                        help="The output file")
    args = parser.parse_args()

    with open(args.positivefile, "r", encoding="latin") as p:
        with open(args.negativefile, "r", encoding="latin") as n:
            with open(args.outfile, "w", encoding="latin") as o:
                MergeDataset()(p, n, o)
