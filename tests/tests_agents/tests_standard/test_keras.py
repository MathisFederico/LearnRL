import pytest
import numpy as np

import learnrl as rl
from learnrl.environments import CatchEnv
from learnrl.agent_parts.estimator import KerasEstimator
from learnrl.agents import StandardAgent

@pytest.mark.slow
def test_keras_pipeline():
    from keras.layers import Conv2D, Flatten, Dense
    from keras.models import Sequential
    from keras.optimizers import Adam

    class MyEstimator(KerasEstimator):

        def build(self):
            self.model = Sequential()
            print(self.observation_size, self.observation_shape)
            self.model.add(Dense(self.observation_size, activation='relu', input_shape=self.observation_shape))
            self.model.add(Dense(self.action_size))

            self.model.compile(Adam(learning_rate=self.learning_rate), loss='mse')

        def preprocess(self, observations, actions):
            return observations

    env = CatchEnv()
    custom_action_value = MyEstimator(observation_space=env.observation_space, action_space=env.action_space, batch_size=64, freezed_steps=20)
    agent = StandardAgent(observation_space=env.observation_space, action_space=env.action_space, action_values=custom_action_value)

    pg = rl.Playground(env, agent)
    pg.fit(1, verbose=1)
