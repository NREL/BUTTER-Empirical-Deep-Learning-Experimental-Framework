from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union
from dataclasses import dataclass

from dmp.task.experiment.training_experiment.run_spec import RunSpec


# ParameterPrimitive = Union[None, bool, int, float, str]
# ParameterValue = Union[ParameterPrimitive, List["ParameterValue"]]
# ParameterDict = Dict[str, "Parameter"]
# Parameter = Union[ParameterValue, ParameterDict]
# FlatParameterDict = Dict[str, ParameterValue]

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dmp.context import Context


@dataclass
class Experiment(ABC):
    data: dict  # extra tags related to this experiment, including batch

    @property
    def version(self) -> int:
        return 100

    @abstractmethod
    def __call__(
        self,
        context: Context,
        run: RunSpec,
    ) -> None:
        pass

    # def get_parameters(self) -> FlatParameterDict:
    #     parameters = self.extract_parameters()
    #     # parameters['task'] = type(self).__name__ # migrated to marshal class type
    #     parameters["task_version"] = self.version
    #     return parameters  # type: ignore

    # def extract_parameters(self) -> FlatParameterDict:
    #     from dmp.marshaling import marshal, flat_marshal_config

    #     separator = "_"
    #     marshaled = marshal.marshal(self, flat_marshal_config)
    #     parameters = {}

    #     def get_parameters(key, target):
    #         target_type = type(target)

    #         # special recursive handling of dicts
    #         if target_type is dict:
    #             # for marshaled and keras objects, generate a (key : type) parameter
    #             # instead of a (key_type_key : type) parameter
    #             for type_key in (marshal_type_key, keras_type_key):
    #                 get_parameters(key, target.pop(type_key))

    #             # recurse into dicts to genreate kvp's from their contents
    #             for k, v in target.items():
    #                 get_parameters(key + separator + k, v)

    #         else:
    #             if key in parameters:
    #                 raise KeyError(f'Parameter conflict on key "{key}".')

    #             parameters[key] = target  # add (key: target) kvp to parameters

    #     # parameters['type'] = marshaled.pop(marshal_type_key)
    #     for k, v in marshaled.items():
    #         get_parameters(k, v)

    #     return parameters
