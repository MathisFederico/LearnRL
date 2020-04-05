# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import learnrl as rl
from learnrl.environments import RdNimEnv, CrossesAndNoughtsEnv
from learnrl.agents.deepRL.agent import DeepRLAgent
from learnrl.agents.deepRL.estimator import Estimator
from learnrl.agents import TableAgent

import numpy as np
from copy import copy

from keras.layers import Conv2D, Flatten, Dense
from keras.models import Model, Sequential
from keras.optimizers import Adam

class KerasEstimator(Estimator):

    def build(self):
        
        self.model = Sequential()
        self.model.add(Flatten(input_shape=self.observation_shape))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.action_size))

        self.model.compile(Adam(learning_rate=0.01), loss='mse')
        # self.model.summary()

    def preprocess(self, observations):
        X = copy(observations)
        X[X==2] = -1
        return X

    def fit(self, observations, actions, Y):

        x_train = self.preprocess(observations)
        
        y_train = self.model.predict(x_train)
        action_ids = self.encoder(actions, arr_type='action')
        y_train[:, action_ids] = Y

        self.model.fit(x_train, y_train, verbose=0)

    def predict(self, observations, actions=None):
        x = self.preprocess(observations)
        Y = self.model.predict(x)

        if actions is not None:
            action_id = self.encoder(actions, arr_type='action')
            return Y[action_id]
        else:
            return Y

env = CrossesAndNoughtsEnv()

custom_action_value = KerasEstimator(observation_space=env.observation_space, action_space=env.action_space)

agent = DeepRLAgent(observation_space=env.observation_space, action_space=env.action_space, action_values=custom_action_value)
agent2 = DeepRLAgent(observation_space=env.observation_space, action_space=env.action_space)

agents = [agent, agent2]
pg = rl.Playground(env, agents)
pg.run(2000, render=False, verbose=1)
