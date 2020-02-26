import numpy as np

from adaptive.Epsilon import Epsilon
from adaptive.LearningRate import LearningRate
from algorithms.TDLearning import TDLearning
from tasks.Task import Task
from agents.DeepAgent import DeepAgent


class DeepQLearning(TDLearning):
        
    def __init__(self, discount, episode_length, encoding):
        super().__init__(discount, episode_length)
        self.encoding = encoding
    
    def run_episode(self, Q : DeepAgent, task : Task, epsilon : Epsilon, alpha : LearningRate):
        
        # to compute backup
        rewards = np.zeros(self.episode_length, dtype=np.float32)
        epsilons = np.zeros(self.episode_length, dtype=np.float32)
        
        # initialize state
        state = task.initial_state()
        encoded_state = self.encoding(state)
        
        # repeat for each step of episode
        for t in range(self.episode_length):
            
            # choose action from state using policy derived from Q
            epsilons[t] = epsilon_t = epsilon.get_epsilon(state)
            action = self.epsilon_greedy(Q, task, encoded_state, epsilon_t)

            # take action and observe reward and new state
            new_state, reward, done = task.transition(state, action) 
            rewards[t] = reward  
            encoded_new_state = self.encoding(new_state)
            
            # compute model means for exploration
            G_Q = reward + self.gamma * Q.max_value(encoded_new_state)
            G_U = reward + self.gamma * np.mean(Q.values(encoded_new_state))  
            
            # update Q
            old_Q = Q.values(encoded_state)[action]
            Q.remember(encoded_state, action, reward, encoded_new_state, done)
            Q.train(self.gamma)
            new_Q = Q.values(encoded_state)[action]
            
            # update epsilon
            epsilon.update_from_experts(state, data=(G_Q, G_U, new_Q - old_Q))
            
            # update state
            state = new_state
            encoded_state = encoded_new_state
            
            # until state is terminal
            if done:
                break
            
        # finish the episode training and return the progress
        Q.finish_episode()            
        return t + 1, rewards[0:t + 1], epsilons[0:t]
    
    def evaluate(self, Q : DeepAgent, task : Task):
        steps = self.episode_length
        rewards = np.zeros(steps, dtype=np.float32)
        state = task.initial_state(training=False)
        # print(state)
        for t in range(steps):
            encoded_state = self.encoding(state)
            action = Q.max_action(encoded_state)
            new_state, reward, done = task.transition(state, action)
            # print('state = {}, reward = {}'.format(new_state, reward))
            rewards[t] = reward
            if done:
                break
            state = new_state
        gamma = self.gamma
        result = 0.0
        for s in range(t, -1, -1):
            result = rewards[s] + gamma * result
        return result, t + 1
