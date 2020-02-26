from collections import defaultdict
import numpy as np
import math

from agents.Agent import Agent


class Tabular(Agent):

    def __init__(self, valid_actions,
                 clip_min=-math.inf, clip_max=math.inf, randomizer=np.zeros):
        self.valid_actions = valid_actions
        self.randomizer = randomizer
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.clear()
        
    def clear(self):
        self.Q = defaultdict(lambda: self.randomizer(self.valid_actions))
    
    def values(self, state):
        return self.Q[state]
     
    def update(self, state, action, error): 
        change = self.alpha * error
        change = max(min(change, self.clip_max), self.clip_min)
        self.Q[state][action] += change
    
