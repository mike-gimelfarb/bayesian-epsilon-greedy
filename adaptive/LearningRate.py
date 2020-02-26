from abc import ABC, abstractmethod


class LearningRate(ABC):
    
    @abstractmethod
    def clear(self):
        pass
    
    @abstractmethod
    def get_alpha(self, state):
        pass
    
    @abstractmethod
    def update_alpha(self, state, data):
        pass
    
    @abstractmethod
    def update_end_of_episode(self, episode):
        pass


class FixedLearningRate(LearningRate):
    
    def __init__(self, value):
        self.value = value
        
    def clear(self):
        pass
    
    def get_alpha(self, state):
        return self.value
    
    def update_alpha(self, state, data):
        pass
    
    def update_end_of_episode(self, episode):
        pass


class ExponentialDecayLearningRate(LearningRate):
    
    def __init__(self, initial, decay, step_decay=1.0, minim=0.0):
        self.initial = initial
        self.decay = decay
        self.step_decay = step_decay
        self.min = minim
        
    def clear(self):
        self.value = self.initial
    
    def get_alpha(self, state):
        return self.value
    
    def update_alpha(self, state, data):
        self.value *= self.step_decay
        self.value = max(self.min, self.value)
    
    def update_end_of_episode(self, episode):
        self.value *= self.decay
        self.value = max(self.min, self.value)
