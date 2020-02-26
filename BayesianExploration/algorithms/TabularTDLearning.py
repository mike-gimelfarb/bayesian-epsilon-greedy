import numpy as np

from agents.Tabular import Tabular
from algorithms.TDLearning import TDLearning
from tasks.Task import Task


class TabularTDLearning(TDLearning):
    
    def evaluate(self, Q : Tabular, task : Task):
        steps = self.episode_length
        rewards = np.zeros(steps, dtype=np.float32)
        state = task.initial_state(training=False)
        # print(state)
        for t in range(steps):
            action = Q.max_action(state)
            new_state, reward, done = task.transition(state, action)
         #   print('state = {}, reward = {}'.format(new_state, reward))
            rewards[t] = reward
            if done:
                break
            state = new_state
        gamma = self.gamma
        result = 0.0
        for s in range(t, -1, -1):
            result = rewards[s] + gamma * result
        return result, t + 1
