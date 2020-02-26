from abc import abstractmethod
import numpy as np

from agents.Agent import Agent


class DeepAgent(Agent):
    
    def __init__(self, state_dim, valid_actions, model_lambda,
                 batch_size=32, epochs=1, memory_size=1000):
        self.state_dim = state_dim
        self.valid_actions = valid_actions
        self.model_lambda = model_lambda
        self.batch_size = batch_size
        self.epochs = epochs
        self.memory_size = memory_size
        
        # placeholder for batch data
        self.states = np.zeros((self.batch_size, self.state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.batch_size, self.state_dim), dtype=np.float32)
        
    def values(self, state):
        return self.model.predict(state)[0]
     
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    @abstractmethod
    def clear(self):
        pass
    
    @abstractmethod
    def train(self, discount):
        pass
    
    @abstractmethod
    def finish_episode(self):
        pass
