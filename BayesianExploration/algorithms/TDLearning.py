from abc import ABC, abstractmethod
import numpy as np
import random
import math

from adaptive.Epsilon import Epsilon
from agents.Agent import Agent
from tasks.Task import Task
from adaptive.LearningRate import LearningRate


class TDLearning(ABC):
   
    def __init__(self, discount, episode_length):
        self.gamma = discount
        self.episode_length = episode_length
    
    @abstractmethod
    def run_episode(self, Q : Agent, task : Task, epsilon : Epsilon, alpha : LearningRate):
        pass    
    
    @abstractmethod
    def evaluate(self, Q : Agent, task : Task):
        pass
        
    def epsilon_greedy(self, Q : Agent, task : Task, state, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(task.valid_actions())
        else:
            return Q.max_action(state)
        
    def train(self, Q : Agent, task : Task, epsilon : Epsilon, alpha : LearningRate, episodes,
              cache_train=True, test_times=1):
        Q.clear()
        epsilon.clear()
        alpha.clear()
        rewards_history = np.zeros(episodes, dtype=np.float32)
        steps_history = np.zeros(episodes, dtype=np.float32)
        episode_epsilon_history = np.zeros(episodes, dtype=np.float32)
        epsilon_history = []
        conseq_200 = 0
        self.episode = 0
        for e in range(episodes):
            steps, rewards, epsilons = self.run_episode(Q, task, epsilon, alpha)
            if cache_train:
                returns = 0.0
                for r in rewards[::-1]:
                    returns = r + self.gamma * returns
            else:
                returns, steps = 0.0, 0.0
                for _ in range(test_times):
                    returns_, steps_ = self.evaluate(Q, task)
                    returns += returns_ / test_times
                    steps += steps_ / test_times                
            rewards_history[e] = returns
            steps_history[e] = steps
            episode_epsilon_history[e] = np.mean(epsilons)
            epsilon_history.append(epsilons)
            if e % 10 == 0:
                print('{} {} {}'.format(episode_epsilon_history[e], returns, steps))
            epsilon.update_end_of_episode(self.episode)
            alpha.update_end_of_episode(self.episode)
            self.episode += 1        
            
            if steps >= 199.99:
                conseq_200 += 1
            else:
                conseq_200 = 0
            # if conseq_200 >= 4:
            #    rewards_history[e:] = rewards_history[e]
            #    steps_history[e:] = steps_history[e]
            #    episode_epsilon_history[e:] = episode_epsilon_history[e]
            #    break
            
        return steps_history, rewards_history, episode_epsilon_history, \
            np.concatenate(epsilon_history, axis=0)
    
    def train_many(self, Q : Agent, task : Task, epsilon : Epsilon, alpha : LearningRate, episodes, trials,
                   cache_train=True, test_times=1):
        average_steps = np.zeros(episodes, dtype=np.float32)
        average_rewards = np.zeros(episodes, dtype=np.float32)
        average_episode_epsilons = np.zeros(episodes, dtype=np.float32)
        std_error_steps = np.zeros((episodes, trials), dtype=np.float32)
        std_error_rewards = np.zeros((episodes, trials), dtype=np.float32)
        std_error_episode_epsilons = np.zeros((episodes, trials), dtype=np.float32)
        average_epsilons = np.zeros(episodes * self.episode_length, dtype=np.float32)
        std_error_epsilons = np.zeros((episodes * self.episode_length, trials), dtype=np.float32)
        min_len = episodes * self.episode_length
        for i in range(trials):
            print('starting trial {}'.format(i))
            steps, rewards, episode_epsilons, epsilons = \
                self.train(Q, task, epsilon, alpha, episodes, cache_train, test_times)
            min_len = min(min_len, epsilons.size)
            average_steps += steps / trials
            average_rewards += rewards / trials
            average_episode_epsilons += episode_epsilons / trials
            std_error_steps[:, i] = steps
            std_error_rewards[:, i] = rewards
            std_error_episode_epsilons[:, i] = episode_epsilons
            average_epsilons = average_epsilons[:min_len] + epsilons[:min_len] / trials
            std_error_epsilons = std_error_epsilons[:min_len, :]
            std_error_epsilons[:, i] = epsilons[:min_len]
            print('ending with {} {}'.format(epsilons[-1], rewards[-1]))
        std_error_steps = np.std(std_error_steps, axis=-1, ddof=1) / math.sqrt(trials)
        std_error_rewards = np.std(std_error_rewards, axis=-1, ddof=1) / math.sqrt(trials)
        std_error_episode_epsilons = np.std(std_error_episode_epsilons, axis=-1, ddof=1) / math.sqrt(trials)
        std_error_epsilons = np.std(std_error_epsilons, axis=-1, ddof=1) / math.sqrt(trials)
        return np.column_stack((average_steps, average_rewards, average_episode_epsilons,
                                std_error_steps, std_error_rewards, std_error_episode_epsilons)), \
            np.column_stack((average_epsilons, std_error_epsilons))
            
