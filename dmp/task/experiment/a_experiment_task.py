from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from dataclasses import dataclass
from dmp.task.task import Task
from dmp.common import keras_type_key, marshal_type_key

ParameterPrimitive = Union[None, bool, int, float, str]
ParameterValue = Union[ParameterPrimitive, List["ParameterValue"]]
ParameterDict = Dict[str, "Parameter"]
Parameter = Union[ParameterValue, ParameterDict]
FlatParameterDict = Dict[str, ParameterValue]


@dataclass
class AExperimentTask(Task, ABC):
    @property
    def version(self) -> int:
        return super().version + 2

    @property
    @abstractmethod
    def batch(self) -> str:
        pass

    @property
    @abstractmethod
    def tags(self) -> Optional[FlatParameterDict]:
        pass

    @property
    @abstractmethod
    def run_tags(self) -> Optional[FlatParameterDict]:
        pass

    def get_parameters(self) -> FlatParameterDict:
        parameters = self.extract_parameters()
        # parameters['task'] = type(self).__name__ # migrated to marshal class type
        parameters["task_version"] = self.version
        return parameters  # type: ignore

    def extract_parameters(self) -> FlatParameterDict:
        from dmp.marshaling import marshal, flat_marshal_config

        separator = "_"
        marshaled = marshal.marshal(self, flat_marshal_config)
        parameters = {}

        def get_parameters(key, target):
            target_type = type(target)

            # special recursive handling of dicts
            if target_type is dict:
                # for marshaled and keras objects, generate a (key : type) parameter
                # instead of a (key_type_key : type) parameter
                for type_key in (marshal_type_key, keras_type_key):
                    get_parameters(key, target.pop(type_key))

                # recurse into dicts to genreate kvp's from their contents
                for k, v in target.items():
                    get_parameters(key + separator + k, v)

            else:
                if key in parameters:
                    raise KeyError(f'Parameter conflict on key "{key}".')

                parameters[key] = target  # add (key: target) kvp to parameters

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
                    plan.append((k, v, dst_prefix + k[len(src_prefix) :], rename))
                    break

        for src_key, v, dst_key, rename in plan:
            if rename:
                del target[src_key]

        for src_key, v, dst_key, rename in plan:
            target[dst_key] = v
        return target
