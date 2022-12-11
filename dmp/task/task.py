from abc import ABC, abstractmethod
import collections as collections
import collections.abc
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Union
from dmp.jobqueue_interface import keras_type_key, marshal_type_key, jobqueue_marshal
from dmp import jobqueue_interface
from dmp.task.task_util import flatten

ParameterValue = Union[None, bool, int, float, str]
ParameterDict = Dict[str, "Parameter"]
Parameter = Union[ParameterValue, ParameterDict]
FlatParameterDict = Dict[str, ParameterValue]


@dataclass
class Task(ABC):

    seed: int
    batch: str

    @abstractmethod
    def __call__(self, worker) -> Dict[str, Any]:
        pass

    @property
    def version(self) -> int:
        return -1

    @property
    def parameters(self) -> ParameterDict:
        parameters = self.extract_parameters()
        # parameters['task'] = type(self).__name__ # migrate to marshal class type
        parameters['task_version'] = self.version 
        return parameters  # type: ignore

    def extract_parameters(self) -> ParameterDict:
        separator = '_'
        marshaled = jobqueue_marshal.marshal(self)
        parameters = {}

        def get_parameters(prefix, target):
            target_type = type(target)
            if target_type is dict:
                skip_key = marshal_type_key
                if marshal_type_key in target:
                    parameters[prefix] = target[marshal_type_key]
                elif keras_type_key in target:
                    parameters[prefix] = target[keras_type_key]
                    skip_key = marshal_type_key

                for k, v in target.items():
                    if k != skip_key:
                        get_parameters(prefix + separator + k, v)
            else:
                parameters[prefix] = target

        get_parameters('', marshaled)
        return parameters
        
    # def extract_parameters(
    #     self,
    #     exclude={},
    #     connector='_',
    # ) -> FlatParameterDict:
        

    #     extracted = {}
    #     path = []

    #     def extract(target, exclude):
    #         nonlocal path, extracted

    #         if target is None or \
    #                 isinstance(target, str) or \
    #                 isinstance(target, int) or \
    #                 isinstance(target, float) or \
    #                 isinstance(target, bool):
    #             p = connector.join(path)
    #             extracted[p] = target
    #             return

    #         if isinstance(target, collections.abc.Mapping):
    #             target = target.items()
    #         elif isinstance(target, collections.abc.Iterable):
    #             raise NotImplementedError()
    #         else:
    #             target = vars(target).items()

    #         for k, v in target:
    #             key_exclude = exclude.get(k, {})
    #             if key_exclude is None:
    #                 continue

    #             path.append(k)
    #             extract(v, key_exclude)
    #             path.pop()

    #     extract(self, exclude)
    #     return extracted
