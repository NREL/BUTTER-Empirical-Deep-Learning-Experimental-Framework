from enum import Enum


class MLTask(str, Enum):
    classification = "classification"
    regression = "regression"
