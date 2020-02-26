import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    

import numpy as np

from agents.DQN import DQN
from agents.Tabular import Tabular
from algorithms.DeepQLearning import DeepQLearning
from algorithms.ExpectedSarsa import ExpectedSarsa

from adaptive.Epsilon import BMCRobust
from adaptive.LearningRate import FixedLearningRate
from tasks.Flags import Flags

from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam

# domain
initial = (4, 0)
num_goals = 5
maze = np.array([[0, 0, 0, 0, 5],
                 [0, 2, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [3, 0, 0, 0, 1],
                 [0, 0, 0, 0, 4]])
domain = Flags(maze, initial)


# creates a neural network for DQN
def model_lambda():
    
    # clear the session
    from keras import backend as K
    K.clear_session()

    model = Sequential()
    model.add(Dense(25, input_dim=sum(maze.shape) + 1 + num_goals))
    model.add(Activation('relu'))
    model.add(Dense(25))
    model.add(Activation('relu'))
    model.add(Dense(4))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer=Adam(0.001))
    return model


# agent
agent = Tabular(domain.valid_actions(),
                randomizer=lambda n: np.random.randn(n) * 0.1) 
# agent = DQN(sum(maze.shape) + 1 + num_goals, 4, model_lambda=model_lambda,
#            batch_size=24, epochs=5, memory_size=2000)

# algorithm
trainer = ExpectedSarsa(discount=0.99, episode_length=200)
# trainer = DeepQLearning(discount=0.99, episode_length=200, encoding=domain.default_encoding)

# run
perf, eps = trainer.train_many(agent, domain,
                                # VDBE(0.5, 1.0 / 4, 0.05),
                               BMCRobust(mu=0, tau=1,
                                         a=500, b=500,
                                         alpha=1, beta=1 + 0.01),
                               FixedLearningRate(0.7),
                               episodes=500, trials=10, cache_train=False)

# save result
np.savetxt('rew_bmcs_sarsa.csv', perf, delimiter=',')
np.savetxt('eps_bmcs_sarsa.csv', eps, delimiter=',')
