# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import numpy as np
from learnrl.estimators import Estimator

class TableEstimator(Estimator):

    def build(self, **kwargs):
        dtype = kwargs.get('dtype', None)
        self.name = 'table'
        self.observation_encoder, self.observation_decoder = self._get_table_encoder_decoder(self.observation_space)
        self.action_encoder, self.action_decoder = self._get_table_encoder_decoder(self.action_space)
        self.table = np.zeros((self.observation_size, self.action_size), dtype=dtype)
    
    def load(self, data):
        self.table = data

    def fit(self, observations, actions, Y):
        delta = Y - self.table[observations, actions]
        self.table[observations, actions] += self.learning_rate * delta

        return {'loss': np.mean(delta**2)}
    
    def predict(self, observations, actions):
        if actions is not None:
            return self.table[observations, actions]
        else:
            return self.table[observations, :]
        
    def __str__(self):
        return str(self.table)

