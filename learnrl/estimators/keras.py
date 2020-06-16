# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from learnrl.estimators import Estimator

import numpy as np
from keras.models import Model, clone_model
import keras.backend as K

class KerasEstimator(Estimator):

    def __init__(self, observation_space, action_space,
                       learning_rate=1e-3, learning_rate_decay=0,
                       batch_size=32, freezed_steps=0, **kwargs):
        self.model = Model()
        super().__init__(observation_space, action_space, learning_rate, learning_rate_decay, **kwargs)
        self.action_encoder, self.action_decoder = self._get_table_encoder_decoder(self.action_space)
        self.freezed_steps = freezed_steps
        if self.freezed_steps > 0:
            self.step_freezed_left = freezed_steps
            self.model_freezed = clone_model(self.model)
        self.name = 'keras'
        self.batch_size = batch_size

    def build(self, **kwargs):
        raise NotImplementedError

    def preprocess(self, observations, actions=None):
        raise NotImplementedError

    def fit(self, observations, actions, Y):
        logs = {}

        x_train = self.preprocess(observations, actions)
        y_train = self.model.predict_on_batch(x_train)
        y_train[np.arange(len(actions)), actions] = Y
        loss = self.model.train_on_batch(x_train, y_train)
        logs.update({'loss': loss})

        if self.freezed_steps > 0:
            logs.update({'freezed_update': self.step_freezed_left == 0})
            if self.step_freezed_left == 0:
                self.model_freezed.set_weights(self.model.get_weights()) 
                self.step_freezed_left = self.freezed_steps
                if self.verbose > 0:
                    print("Freezed model updated")
            self.step_freezed_left -= 1
        
        self.update_learning_rate(K.eval(self.model.optimizer.lr))
        return logs

    def predict(self, observations, actions=None):
        x = self.preprocess(observations, actions)
        if self.freezed_steps > 0:
            Y = self.model_freezed.predict_on_batch(x)
        else:
            Y = self.model.predict_on_batch(x)
        if actions is not None:
            return Y[actions]
        else:
            return Y

