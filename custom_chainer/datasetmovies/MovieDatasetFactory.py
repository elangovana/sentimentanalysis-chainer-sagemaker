from datasetmovies.MovieDatasetIterator import MovieDatasetIterator
from datasetmovies.MovieDatasetIteratorProcessor import MovieDatasetIteratorProcessor


class MovieDatasetFactory:

    @property
    def name(self):
        return "movie"

    def get_dataset(self, path):
        iterator = MovieDatasetIterator(path, has_header=False)
        data_processor = MovieDatasetIteratorProcessor(iterator)
        return data_processor