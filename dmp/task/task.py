from abc import ABC, abstractmethod
import collections as collections
from dataclasses import dataclass
import os
import platform
import subprocess
from typing import Dict, List, Tuple, Union

import tensorflow

ParameterValue = Union[bool, int, float, str]
ParameterDict = Dict[str, "Parameter"]
Parameter = Union[ParameterValue, ParameterDict]
FlatParameterDict = Dict[str, ParameterValue]


@dataclass
class Task(ABC):

    seed: int
    batch: str

    @abstractmethod
    def __call__(self, *args, **kwargs
                 # TODO: fix the history part of this return type
                 ) -> Tuple[ParameterDict, Dict[str, any]]:
        pass

    @property
    def version(self) -> int:
        return -1

    @property
    def parameters(self) -> ParameterDict:
        parameters = self.extract_parameters()
        parameters['task'] = type(self).__name__

        # def extract_keys(src, keys):
        #     dest = {}
        #     for k in keys:
        #         if k in src:
        #             dest[k] = src[k]
        #             del src[k]
        #     return dest

        # run_parameters = extract_keys(
        #     experiment_parameters, self._run_parameter_keys)
        # run_values = extract_keys(
        #     experiment_parameters, self._run_value_keys)

        parameters['task_version'] = self.version # new
        parameters['tensorflow_version'] = str(tensorflow.__version__) # new
        parameters['python_version'] = str(platform.python_version())  # new
        parameters['platform'] = str(platform.platform())  # new

        git_hash = None
        try:
            git_hash = subprocess.check_output(
                ["git", "describe", "--always"],
                cwd=os.path.dirname(__file__)).strip().decode()
        except:
            pass
        parameters['git_hash'] = git_hash # new

        parameters['hostname'] = str(platform.node()) # new
        parameters['slurm_job_id'] = os.getenv("SLURM_JOB_ID") # new
 
        return parameters

    def extract_parameters(
        self,
        exclude={},
        connector='.',
    ) -> FlatParameterDict:
        extracted = {}
        path = []

        def extract(target, exclude):
            nonlocal path, extracted

            if target is None or \
                    isinstance(target, str) or \
                    isinstance(target, int) or \
                    isinstance(target, float) or \
                    isinstance(target, bool):
                p = connector.join(path)
                extracted[p] = target
                return

            if isinstance(target, collections.Mapping):
                target = target.items()
            elif isinstance(target, collections.Iterable):
                raise NotImplementedError()
            else:
                target = vars(target).items()

            for k, v in target:
                key_exclude = exclude.get(k, {})
                if key_exclude is None:
                    continue

                path.append(k)
                extract(v, key_exclude)
                path.pop()

        extract(self, exclude)
        return extracted
