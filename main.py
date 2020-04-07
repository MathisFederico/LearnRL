# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import learnrl as rl
from learnrl.environments import RdNimEnv, CrossesAndNoughtsEnv
from learnrl.agents.deepRL.agent import DeepRLAgent
from learnrl.agents.deepRL.estimator import Estimator, KerasEstimator
from learnrl.agents import TableAgent

import numpy as np
from copy import copy

from keras.layers import Conv2D, Flatten, Dense
from keras.models import Model, Sequential
from keras.optimizers import Adam

class MyEstimator(KerasEstimator):

    def build(self):
        
        self.model = Sequential()
        self.model.add(Flatten(input_shape=self.observation_shape))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.action_size))

        self.model.compile(Adam(learning_rate=0.01), loss='mse')
        # self.model.summary()

    def preprocess(self, observations, actions):
        X = copy(observations)
        X[X==2] = -1
        return X


env = CrossesAndNoughtsEnv()

custom_action_value = MyEstimator(observation_space=env.observation_space, action_space=env.action_space)

agent = DeepRLAgent(observation_space=env.observation_space, action_space=env.action_space)
agent2 = DeepRLAgent(observation_space=env.observation_space, action_space=env.action_space)

agents = [agent, agent2]
pg = rl.Playground(env, agents)
pg.run(100000, render=False, verbose=1)
