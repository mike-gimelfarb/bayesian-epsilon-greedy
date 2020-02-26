import random
import numpy as np
from collections import deque

from agents.DeepAgent import DeepAgent


class DQN(DeepAgent):

    def __init__(self, state_dim, valid_actions, model_lambda,
                 batch_size=32, epochs=1, memory_size=1000):
        super().__init__(state_dim, valid_actions, model_lambda, batch_size, epochs, memory_size)
        
    def clear(self):
        self.model = self.model_lambda()
        self.memory = deque(maxlen=self.memory_size)
        
    def train(self, discount):

        # we don't have enough memory for training
        if len(self.memory) < self.batch_size: return

        # sample a minibatch
        batch_size = self.batch_size
        mini_batch = random.sample(self.memory, batch_size)
        states = self.states
        next_states = self.next_states
        for i, memory in enumerate(mini_batch):
            state, _, _, new_state, _ = memory
            states[i] = state
            next_states[i] = new_state

        # make the predictions with current models
        values = self.model.predict(states)
        next_values = self.model.predict(next_states)

        # update model weights based on error in prediction
        for i, memory in enumerate(mini_batch):
            _, action, reward, _, done = memory
            values[i][action] = reward
            if not done:
                values[i][action] += discount * np.amax(next_values[i])

        # make batch which includes target q value and predicted q value
        self.model.fit(states, values,
                       batch_size=batch_size,
                       epochs=self.epochs,
                       verbose=0)
    
    def finish_episode(self):
        pass
