from abc import abstractmethod


class Task:

    @property
    @abstractmethod
    def type(self) -> TaskType: