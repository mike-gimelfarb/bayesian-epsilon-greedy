from abc import ABC, abstractmethod
import numpy as np


class Agent(ABC):
    
    @abstractmethod
    def clear(self):
        pass
    
    @abstractmethod
    def values(self, state):
        pass
    
    def max_action(self, state):
        return np.argmax(self.values(state))
    
    def max_value(self, state):
        return np.amax(self.values(state))
    
    
