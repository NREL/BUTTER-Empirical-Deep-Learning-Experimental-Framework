from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
)
from dataclasses import dataclass

from dmp.dataset.dataset import Dataset
from dmp.dataset.dataset_group import DatasetGroup
from dmp.dataset.dataset_loader import DatasetLoader, dataset_cache_directory

import copy

from dmp.dataset.ml_task import MLTask 

class Foo:
    last : Any = ('', None)

class PMLBDatasetLoader(DatasetLoader):
    '''
    Loads PMLB datasets
    see: https://github.com/EpistasisLab/pmlb
    https://epistasislab.github.io/pmlb/
    '''
    def __init__(
        self,
        dataset_name: str,
        ml_task: MLTask,
    ):
        super().__init__('pmlb', dataset_name, ml_task)
    # def _load_dataset(self):
    #     return self._fetch_from_source()

    def _fetch_from_source(self):
        import pmlb

        if Foo.last[0] == self.dataset_name:
            d = Foo.last[1]
        else:
            d = pmlb.fetch_data(
                            self.dataset_name,
                            return_X_y=True,
                            local_cache_dir=dataset_cache_directory,
                        )
            Foo.last = (self.dataset_name, d)
        d = copy.deepcopy(d)
        
        return Dataset(self.ml_task,
                       DatasetGroup(*d))  # type: ignore

    # def _get_cache_path(self, name):
    #     return os.path.join(self.dataset_cache_directory, self.dataset_name, name)

    # def _try_read_from_cache(self) -> Optional[Dataset]:
    #     filename = self._get_cache_path('.npy')
    #     try:
    #         os.makedirs(self.dataset_cache_directory, exist_ok=True)
    #         dataset = pandas.read_csv(filename, sep='\t', compression='gzip')
    #         X = dataset.drop('target', axis=1).values
    #         y = dataset['target'].values
    #         return Dataset(self.ml_task, DatasetGroup(X, y))
    #     except FileNotFoundError:
    #         return None
    #     except:
    #         print(f'Error reading from dataset cache for {self}:')
    #         traceback.print_exc()
    #         try:
    #             os.remove(filename)
    #         except:
    #             print(f'Error removing bad cache file for {self}:')
    #             traceback.print_exc()
    #     return None
