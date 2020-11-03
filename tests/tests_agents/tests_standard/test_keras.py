import pytest
import numpy as np

import learnrl as rl
from learnrl.environments import CatchEnv
from learnrl.estimators.tensorflow import KerasEstimator
from learnrl.agents import StandardAgent

from gym import spaces

import importlib
tensorflow_spec = importlib.util.find_spec('tensorflow')

if tensorflow_spec is not None:
    from learnrl.estimators import KerasEstimator
    import tensorflow as tf

    @pytest.mark.slow
    def test_fit():
        class MyEstimator(KerasEstimator):

            def build(self):
                self.model = tf.keras.experimental.LinearModel(units=self.action_size, use_bias=False)
                self.model.compile(tf.keras.optimizers.SGD(learning_rate=self.learning_rate), loss='mse')

            def preprocess(self, observations, actions):
                return tf.one_hot(observations, self.observation_size, on_value=1, off_value=0)
        
        observation_size, action_size = 3, 2
        observations = np.array([0, 2, 0, 1])
        actions = np.array([0, 0, 1, 1])
        returns = np.array([1, -1, 1, -1], dtype=np.float32)

        estimator = MyEstimator(spaces.Discrete(observation_size), spaces.Discrete(action_size), learning_rate=len(observations))
        estimator.fit(observations, actions, returns)

        expected_weights = np.array([[1, 1], [0, -1],[-1, 0]], dtype=np.float32)
        new_returns = estimator.predict(observations).numpy()[np.arange(len(actions)), actions]

        print(new_returns, returns)
        assert np.allclose(new_returns, returns)
        print(estimator.model.weights[0].numpy(), expected_weights)
        assert np.allclose(estimator.model.weights[0].numpy(), expected_weights)

    @pytest.mark.slow
    def test_keras_pipeline():
        class MyEstimator(KerasEstimator):

            def build(self):
                self.model = tf.keras.models.Sequential()
                self.model.add(tf.keras.layers.Dense(self.observation_size, activation='relu', input_shape=self.observation_shape))
                self.model.add(tf.keras.layers.Dense(self.action_size))

                self.model.compile(tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')

            def preprocess(self, observations, actions):
                return observations

        env = CatchEnv()
        custom_action_value = MyEstimator(observation_space=env.observation_space, action_space=env.action_space, batch_size=64, freezed_steps=20)
        agent = StandardAgent(observation_space=env.observation_space, action_space=env.action_space, action_values=custom_action_value)

        pg = rl.Playground(env, agent)
        pg.fit(1)
