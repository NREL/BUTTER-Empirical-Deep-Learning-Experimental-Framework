from abc import ABC, abstractmethod
import collections as collections
import collections.abc
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type, Union

from jobqueue.job import Job
from dmp.common import keras_type_key, marshal_type_key, tensorflow_type_key, tensorflow_config_key
from dmp.task.task_result_record import TaskResultRecord
from dmp.worker import Worker

ParameterValue = Union[None, bool, int, float, str]
ParameterDict = Dict[str, 'Parameter']
Parameter = Union[ParameterValue, ParameterDict]
FlatParameterDict = Dict[str, ParameterValue]

# # register task types here
# # task_types: List['Task'] = []
# def register_task_type(type: Type) -> None:
#     # task_types.append(type)
#     from dmp.marshal_registry import register_type
#     register_type(type)


@dataclass
class Task(ABC):

    seed: int
    batch: str

    @abstractmethod
    def __call__(self, worker: Worker, job: Job) -> TaskResultRecord:
        pass

    @property
    def version(self) -> int:
        return -1

    def get_parameters(self) -> ParameterDict:
        parameters = self.extract_parameters()
        # parameters['task'] = type(self).__name__ # migrate to marshal class type
        parameters['task_version'] = self.version
        return parameters  # type: ignore

    # def extract_parameters(self) -> ParameterDict:
    #     from dmp.jobqueue_interface import jobqueue_marshal
    #     return jobqueue_marshal.marshal(self)

    def extract_parameters(self) -> ParameterDict:
        from dmp.marshaling import marshal
        separator = '_'
        marshaled = marshal.marshal(self, circular_references_only=True)
        parameters = {}

        def get_parameters(key, target):
            target_type = type(target)
            if target_type is dict:
                if marshal_type_key in target:
                    get_parameters(key, target.pop(marshal_type_key))

                if keras_type_key in target:
                    get_parameters(key, target.pop(keras_type_key))
                    # parameters[key] = target.pop(keras_type_key)
                # if tensorflow_type_key in target:
                #     parameters[key] = target[tensorflow_type_key]
                #     target = target.get(tensorflow_config_key, {})

                for k, v in target.items():
                    get_parameters(key + separator + k, v)
            else:
                if key in parameters:
                    raise KeyError(f'Parameter conflict on key "{key}".')
                parameters[key] = target

        # parameters['type'] = marshaled.pop(marshal_type_key)
        for k, v in marshaled.items():
            get_parameters(k, v)

        return parameters
