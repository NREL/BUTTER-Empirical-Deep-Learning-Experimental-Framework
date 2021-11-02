import itertools
from attr import dataclass

@dataclass
class CartesianBatch():

    def __task_class(self):
        """
        Return the class for which tasks will be instantiated.
        """
        pass

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
