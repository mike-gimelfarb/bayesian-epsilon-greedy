import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    

import numpy as np

from agents.DQN import DQN
from agents.Tabular import Tabular
from algorithms.ExpectedSarsa import ExpectedSarsa
from algorithms.DeepQLearning import DeepQLearning

from adaptive.Epsilon import VDBE, BMCRobust
from adaptive.LearningRate import FixedLearningRate
from tasks.Inventory import Inventory

from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.optimizers import Adam

# domain
domain = Inventory()


# creates a neural network for DQN
def model_lambda():
     
    # clear the session
    from keras import backend as K
    K.clear_session()
 
    model = Sequential()
    model.add(Dense(100, input_dim=domain.number_states()))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(domain.valid_actions()))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer=Adam(0.001))
    return model


# agent
agent = Tabular(domain.valid_actions(),
                randomizer=lambda n: np.random.randn(n) * 0.1) 
agent = DQN(domain.number_states(), domain.valid_actions(), model_lambda=model_lambda,
            batch_size=64, epochs=1, memory_size=3000)

# algorithm
trainer = ExpectedSarsa(discount=0.95, episode_length=200)
trainer = DeepQLearning(discount=0.95, episode_length=200, encoding=domain.default_encoding)

# run
perf, eps = trainer.train_many(agent, domain,
                               # Fixed(0.5),
                               # ExponentialDecay(0.5, 0.85),
                               # VDBE(0.5, 1.0 / domain.valid_actions(), 0.01),
                               # BMC(1.0, 1.0 / 0.499999 - 1.0, 100.0),
                               BMCRobust(mu=0, tau=1,
                                         a=500, b=500,
                                         alpha=25, beta=25 + 0.01),
                               FixedLearningRate(0.6),
                               episodes=500, trials=10, cache_train=False, test_times=10)

# trainer.evaluate(agent, domain)

# save result
np.savetxt('rew_vdbe_001_dqn.csv', perf, delimiter=',')
np.savetxt('eps_vdbe_001_dqn.csv', eps, delimiter=',')
