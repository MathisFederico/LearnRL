import pytest
import numpy as np

import learnrl as rl
from learnrl.environments import CatchEnv
from learnrl.estimators import KerasEstimator
from learnrl.agents import StandardAgent

from gym import spaces

def test_fit():
    
    class LinearModel():

        def __init__(self, observation_size, action_size, learning_rate):
            self.weights = np.zeros((observation_size, action_size))
            self.learning_rate = learning_rate

        def predict(self, observations):
            return np.dot(observations, self.weights)

        def fit(self, x, y, **kwargs):
            y_pred = self.predict(x)
            self.weights -= self.learning_rate * np.dot(np.transpose(x), (y_pred - y))

    class DummyEstimator(KerasEstimator):

        def build(self):
            self.model = LinearModel(self.observation_size, self.action_size, self.learning_rate)

        def preprocess(self, observations, actions):
            return np.eye(self.observation_size)[observations]
    
    observation_size, action_size = 3, 2
    estimator = DummyEstimator(spaces.Discrete(observation_size), spaces.Discrete(action_size), learning_rate=1)

    observations = np.array([0, 2, 0, 1])
    actions = np.array([0, 0, 1, 1])
    returns = np.array([1, -1, 1, -1])
    estimator.fit(observations, actions, returns)

    expected_weights = np.array([[1, 1], [0, -1],[-1, 0]])
    new_returns = estimator.predict(observations)[np.arange(len(actions)), actions]

    assert np.allclose(new_returns, returns)
    assert np.allclose(estimator.model.weights, expected_weights)


@pytest.mark.slow
def test_keras_pipeline():
    from keras.layers import Conv2D, Flatten, Dense
    from keras.models import Sequential
    from keras.optimizers import Adam

    class MyEstimator(KerasEstimator):

        def build(self):
            self.model = Sequential()
            self.model.add(Dense(self.observation_size, activation='relu', input_shape=self.observation_shape))
            self.model.add(Dense(self.action_size))

            self.model.compile(Adam(learning_rate=self.learning_rate), loss='mse')

        def preprocess(self, observations, actions):
            return observations

    env = CatchEnv()
    custom_action_value = MyEstimator(observation_space=env.observation_space, action_space=env.action_space, batch_size=64, freezed_steps=20)
    agent = StandardAgent(observation_space=env.observation_space, action_space=env.action_space, action_values=custom_action_value)

    pg = rl.Playground(env, agent)
    pg.fit(1)
