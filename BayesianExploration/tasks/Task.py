from abc import ABC, abstractmethod


class Task(ABC):

    @abstractmethod
    def initial_state(self, training=True):
        pass

    @abstractmethod
    def valid_actions(self):
        pass

    @abstractmethod
    def transition(self, state, action):
        pass
    
