# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from learnrl.estimators import Estimator

import numpy as np
from tensorflow.keras.models import Model, clone_model
import tensorflow as tf

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

    def preprocess(self, observations:tf.Tensor, actions:tf.Tensor=None):
        raise NotImplementedError

    def fit(self, observations, actions, Y):
        logs = {}

        observations_tensor = tf.convert_to_tensor(observations)
        actions_tensor = tf.convert_to_tensor(actions)
        Y_tensor = tf.convert_to_tensor(Y, dtype=tf.float32)

        metrics = self.train_on_experience_batch(observations_tensor, actions_tensor, Y_tensor)
        metrics = {key:metrics[key].numpy() for key in metrics}
        logs.update(metrics)

        if self.freezed_steps > 0:
            logs.update({'freezed_update': self.step_freezed_left == 0})
            if self.step_freezed_left == 0:
                self.model_freezed.set_weights(self.model.get_weights())
                self.step_freezed_left = self.freezed_steps
                if self.verbose > 0:
                    print("Freezed model updated")
            self.step_freezed_left -= 1

        return logs

    # @tf.function(experimental_relax_shapes=True)
    def train_on_experience_batch(self, observations:tf.Tensor, actions:tf.Tensor, Y:tf.Tensor):
        x_train = self.preprocess(observations, actions)
        y_pred = self.model.predict_step(x_train)

        actions_indices = tf.stack((tf.range(len(actions), dtype=actions.dtype), actions), axis=-1)
        y_action_pred = tf.gather_nd(y_pred, actions_indices)
        y_train = tf.tensor_scatter_nd_add(y_pred, actions_indices, tf.add(Y, -y_action_pred))

        metrics = self.model.train_step((x_train, y_train))
        return metrics

    def predict(self, observations, actions=None):
        x = self.preprocess(observations, actions)
        x = tf.convert_to_tensor(x)
        Y = self.predict_on_experience_batch(x)
        if actions is not None:
            return Y[actions]
        else:
            return Y

    # @tf.function(experimental_relax_shapes=True)
    def predict_on_experience_batch(self, x):
        if self.freezed_steps > 0:
            Y = self.model_freezed.predict_step(x)
        else:
            Y = self.model.predict_step(x)
        return Y
