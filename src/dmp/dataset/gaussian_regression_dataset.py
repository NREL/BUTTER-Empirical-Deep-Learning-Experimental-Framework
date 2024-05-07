import numpy
from dmp.dataset.dataset import Dataset
from dmp.dataset.dataset_group import DatasetGroup
from dmp.dataset.dataset_loader import DatasetLoader
from dmp.dataset.ml_task import MLTask


class GaussianRegressionDataset(DatasetLoader):
    def __init__(self, num_dimensions, scale, num_samples):
        super().__init__(
            "synthetic",
            f"GaussianRegressionDataset_{num_dimensions}_{int(scale*100)}",
            MLTask.regression,
        )
        self.scale = scale
        self.num_dimensions = num_dimensions
        self.num_samples = num_samples

    def _load_dataset(self):
        m = numpy.random.normal(0, 1, size=(self.num_dimensions,))
        # b = numpy.random.normal(0, 1, size=(self.num_dimensions))
        y = numpy.random.uniform(-1, 1, size=(self.num_samples,))
        x = []
        for i in range(self.num_samples):
            r = m * y[i]
            print(r.shape, r)
            x.append(r)

        x = numpy.vstack(x)
        x += numpy.random.normal(0, self.scale, size=x.shape)

        print(x.shape, y.shape)

        return Dataset(self.ml_task, DatasetGroup(x, y))

    def _fetch_from_source(self) -> Dataset:
        return None
