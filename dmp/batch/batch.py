import itertools
from attr import dataclass

from dmp.task.aspect_test.aspect_test_task import AspectTestTask


# TODO: fix this
@dataclass
class CartesianBatch():

    def __task_class(self):
        """
        Return the class for which tasks will be instantiated.
        """

    def __iter__(self):
        """
        batches:
            Implements a cross product over all fields in this dataclass.
            Returns an iterator which yeilds self.__task_class objects which are initialized based on the batch_config.

        returns:
            generator[AspectTestTask]
        """
        batch_description_dict = self.asdict()
        keys = batch_description_dict.keys()
        for vals in itertools.product(*batch_description_dict.vals()):
            yield self._task()(**dict(zip(keys, vals)))

@dataclass
class AspectTestTaskCartesianBatch(CartesianBatch):
    """
    for task in AspectTestTaskCartesianBatch(**config)...
    """
    dataset: List[str] = ['529_pollen'],
    learning_rate: List[float] = [0.001],
    topology: List[str] = ['wide_first'],
    budget: List[int] = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                         32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304,
                         8388608, 16777216, 33554432],
    depth: List[int] = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20]

    def __task_class():
        return AspectTestTask