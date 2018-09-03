from datasetyelp.YelpChainerDataset import YelpChainerDataset
from datasetyelp.YelpReviewDatasetProcessor import YelpReviewDatasetProcessor


class YelpDatasetFactory:

    @property
    def name(self):
        return "yelp"

    def get_dataset(self, path):
        iterator = YelpChainerDataset(path, has_header=True)
        data_processor = YelpReviewDatasetProcessor(iterator)
        return data_processor