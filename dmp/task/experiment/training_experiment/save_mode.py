from abc import ABC, abstractmethod

class SaveMode(ABC):
    '''
    Configures the model saving policy during an experiment.
    '''

    @abstractmethod
    def make_callback(self):
        pass