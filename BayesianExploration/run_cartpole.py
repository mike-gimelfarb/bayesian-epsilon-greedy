import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  

import numpy as np

from agents.DQN import DQN
from agents.Tabular import Tabular
from algorithms.ExpectedSarsa import ExpectedSarsa
from algorithms.DeepQLearning import DeepQLearning
from algorithms.QLearning import QLearning
from tasks.CartPole import CartPole  

from adaptive.Epsilon import BMCRobust
from adaptive.LearningRate import ExponentialDecayLearningRate

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.regularizers import l2

# domain
domain = CartPole(discrete=True)


# creates a neural network for DQN
def model_lambda():
    
    # clear the session
    from keras import backend as K
    K.clear_session()

    model = Sequential()
    model.add(Dense(12, input_dim=4,
                    activation='relu',
                    kernel_regularizer=l2(1e-6)))
    model.add(Dense(12,
                    activation='relu',
                    kernel_regularizer=l2(1e-6)))
    model.add(Dense(2,
                    activation='linear',
                    kernel_regularizer=l2(1e-6)))
    model.compile(loss='mse', optimizer=Adam(0.0005))
    return model


# agent
agent = Tabular(domain.valid_actions(),
                randomizer=lambda n: np.random.randn(n) * 0.0) 
# agent = DQN(4, 2, model_lambda=model_lambda,
#            batch_size=32, epochs=3, memory_size=2000)

# algorithm
trainer = ExpectedSarsa(discount=0.95, episode_length=200)
# trainer = DeepQLearning(discount=0.95, episode_length=200, encoding=domain.default_encoding)

# run
perf, eps = trainer.train_many(agent, domain,
                               # Fixed(0.5),
                               # ExponentialDecay(0.5, 0.99),
                               # PowerLawDecay(0.5, 1.5),
                               # BMCRobustState(mu=0, tau=1,
                               #               a=1, b=1,
                                #              alpha=5, beta=5 + 0.01),
                               BMCRobust(mu=0, tau=1,
                                         a=500, b=500,
                                         alpha=5, beta=5 + 0.01),  # 10 for sarsa, 5 dqn
                               # VDBE(0.5, 1.0 / 2, 0.1, transform=domain.discretize),
                               ExponentialDecayLearningRate(0.5, 0.99, minim=0.01),
                               episodes=300, trials=100, cache_train=False, test_times=10)

# save result
np.savetxt('rew_bmcs_sarsa.csv', perf, delimiter=',')
np.savetxt('eps_bmcs_sarsa.csv', eps, delimiter=',')
