from abc import abstractmethod, ABC


class Task(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs) -> any:
        pass
