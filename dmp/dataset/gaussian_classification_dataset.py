



import numpy
from dmp.dataset.dataset import Dataset
from dmp.dataset.dataset_group import DatasetGroup
from dmp.dataset.dataset_loader import DatasetLoader
from dmp.dataset.ml_task import MLTask


class GaussianClassificationDataset(DatasetLoader):
    

    def __init__(self, num_classes, num_dimensions, scale, num_samples):
        super().__init__('synthetic', f'GaussianClassificationDataset_{num_classes}_{num_dimensions}_{int(scale*100)}',MLTask.classification)
        self.num_classes = num_classes
        self.scale = scale
        self.num_dimensions = num_dimensions
        self.num_samples = num_samples

    def _load_dataset(self):
        classes = numpy.random.normal(0, 1, size=(self.num_classes, self.num_dimensions))
        x = []
        y = []

        for i in range(self.num_samples):
            ci = numpy.random.randint(self.num_classes)
            y.append(ci)
            mu = classes[ci]
            x.append(numpy.random.normal(mu, self.scale, size=(self.num_dimensions)))
        return Dataset(self.ml_task, DatasetGroup(numpy.vstack(x), numpy.vstack(y)))

    def _fetch_from_source(self) -> Dataset:
        return None
