from abc import ABC, abstractmethod
from typing import Dict, Iterable, Mapping, Union
from dmp.logging.result_logger import ResultLogger

from dmp.task.task import Task


class PostgresResultLogger(ResultLogger):

    def log(
        self,
        # TODO: make sure you merged the runtime_parameters in here!
        experiment_parameters: Dict,
        run_parameters: Dict,
        result: Dict,
    ) -> None:
        experiment_parameter_ids = \
            self.get_parameter_ids(experiment_parameters)

        # get experiment_id
        # if experiment doesn't exist, create it using experiment_parameters

        # insert run_parameters and result into run table

        pass

    def get_parameter_ids(self, parameters):
        return [self.get_parameter_id(k, v) for k, v in parameters.items()]

    def get_parameter_id(self, key, value):
        # get parameter id from table
        # if it doesn't exist, create it
        pass
