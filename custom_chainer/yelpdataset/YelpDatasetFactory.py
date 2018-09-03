from yelpdataset.YelpChainerDataset import YelpChainerDataset
from yelpdataset.YelpReviewDatasetProcessor import YelpReviewDatasetProcessor


class YelpDatasetFactory:

    @property
    def name(self):
        return "yelp"

    def get_dataset(self, path):
        iterator = YelpChainerDataset(path, has_header=True)
        data_processor = YelpReviewDatasetProcessor(iterator)
        return data_processor