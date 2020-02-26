import gym
import math
import numpy as np

from tasks.Task import Task


class CartPole(Task):

    def __init__(self, discrete=False):
        self.env = gym.make('CartPole-v1')
        self.buckets = (3, 3, 6, 3)
        self.discrete = discrete

    def initial_state(self, training=True):
        state = self.env.reset()
        if self.discrete:
            state = self.discretize(state)
        return state

    def valid_actions(self):
        return 2

    def transition(self, state, action):
        new_state, _, done, _ = self.env.step(action)
        reward = 1.0 if not done else 0.0
        if self.discrete:
            new_state = self.discretize(new_state)
        return new_state, reward, done

    def render(self, policy, encoder):
        act = self.act
        state = self.initial_state()
        done = False
        while not done:
            self.env.render()
            state_enc = encoder(state)
            action = policy(state_enc)
            new_state, _, done = act(state, action)
            state = new_state

    def default_encoding(self, state):
        res = np.reshape(state, (1, -1))
        return res

    def discretize(self, obs):
        upper_bounds = [self.env.observation_space.high[0], 0.5,
                        self.env.observation_space.high[2], math.radians(50)]
        lower_bounds = [self.env.observation_space.low[0], -0.5,
                        self.env.observation_space.low[2], -math.radians(50)]
        ratios = [(obs[i] + abs(lower_bounds[i])) / 
                  (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
        new_obs = [int(round((self.buckets[i] - 1) * ratios[i])) for i in range(len(obs))]
        new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i])) for i in range(len(obs))]
        return tuple(new_obs)
