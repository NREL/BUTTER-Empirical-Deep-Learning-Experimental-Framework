
from abc import ABC
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from dataclasses import dataclass
from dmp.task.task import Task
from dmp.common import keras_type_key, marshal_type_key

ParameterValue = Union[None, bool, int, float, str]
ParameterDict = Dict[str, 'Parameter']
Parameter = Union[ParameterValue, ParameterDict]
FlatParameterDict = Dict[str, ParameterValue]

@dataclass
class ExperimentTask(Task, ABC):
    seed: int
    batch: str
    tags: Optional[Dict[str, ParameterValue]]
    run_tags: Optional[Dict[str, ParameterValue]]

    def get_parameters(self) -> ParameterDict:
        parameters = self.extract_parameters()
        # parameters['task'] = type(self).__name__ # migrate to marshal class type
        parameters['task_version'] = self.version
        return parameters  # type: ignore

    # def extract_parameters(self) -> ParameterDict:
    #     from dmp.jobqueue_interface import jobqueue_marshal
    #     return jobqueue_marshal.marshal(self)

    def extract_parameters(self) -> ParameterDict:
        from dmp.marshaling import marshal, flat_marshal_config
        separator = '_'
        marshaled = marshal.marshal(self, flat_marshal_config)
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

    @staticmethod
    def remap_key_prefixes(
        target: Dict[str, Any],
        prefix_mapping: Iterable[Tuple[str, str, bool]],
    ) -> dict:
        plan = []
        for k, v in target.items():
            for src_prefix, dst_prefix, rename in prefix_mapping:
                if k.startswith(src_prefix):
                    plan.append((k, v, dst_prefix + k[len(src_prefix):], rename))
                    break

        for src_key, v, dst_key, rename in plan:
            if rename:
                del target[src_key]

        for src_key, v, dst_key, rename in plan:
            target[dst_key] = v
        return target