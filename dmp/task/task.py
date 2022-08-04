from abc import ABC, abstractmethod
import collections as collections
import collections.abc
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
class Task:
    
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

        parameters['task_version'] = self.version
        parameters['tensorflow_version'] = str(tensorflow.__version__)
        parameters['python_version'] = str(platform.python_version()) 
        parameters['platform'] = str(platform.platform()) 

        git_hash = None
        try:
            git_hash = subprocess.check_output(
                ["git", "describe", "--always"],
                cwd=os.path.dirname(__file__)).strip().decode()
        except:
            pass
        parameters['git_hash'] = git_hash

        parameters['hostname'] = str(platform.node())
        parameters['slurm_job_id'] = os.getenv("SLURM_JOB_ID")
 
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

            if isinstance(target, collections.abc.Mapping):
                target = target.items()
            elif isinstance(target, collections.abc.Iterable):
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
