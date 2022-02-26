from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass
class Task(ABC):

    id: int
    seed: int

    @abstractmethod
    def __call__(self, *args, **kwargs) -> any:
        pass

    @property
    def parameters(self):
        experiment_parameters = Task.extract_parameters(self)
        experiment_parameters['task'] = type(self).__name__

        run_parameters = {}
        for k in self._run_parameters:
            run_parameters[k] = experiment_parameters[k]
            del experiment_parameters[k]

        return experiment_parameters, run_parameters

    @property
    def _run_parameters(self):
        return ['seed', 'id']

    @staticmethod
    def extract_parameters(target, exclude={}, connector='.'):
        extracted = {}
        path = []

        def extract(target, exclude):
            nonlocal path, extracted

            if target is None or \
                    isinstance(target, str) or \
                    isinstance(target, int) or \
                    isinstance(target, float) or \
                    isinstance(target, bool) or \
                    isinstance(target, Iterable):
                p = connector.join(path)
                extracted[p] = target
                return

            if isinstance(target, Mapping):
                target = target.items()
            else:
                target = vars(target)

            for k, v in target:
                key_exclude = exclude.get(k, {})
                if key_exclude is None:
                    continue

                path.append(k)
                extract(v, key_exclude)
                path.pop()

        extract(target, exclude, connector)
        return extracted
