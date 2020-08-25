import learnrl as rl
from learnrl.agents import StandardAgent, RandomAgent
from learnrl.estimators.tensorflow import KerasEstimator
# from learnrl.callbacks.tensorflow import TensorboardCallback
import gym

import numpy as np

from tensorflow.keras import layers, optimizers, Input
from tensorflow.keras.models import Sequential

# Load the environment
env = gym.make('CartPole-v1')

# Define the model for our Estimator
class ActionValues(KerasEstimator):
    def preprocess(self, observations, actions):
        return observations

    def build(self, **kwargs):
        print(self.observation_size)
        print(self.action_size)

        model = Sequential()
        model.add(Input(shape=(self.observation_size,)))
        model.add(layers.Dense(8, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros'))
        model.add(layers.Dense(self.action_size, kernel_initializer='random_normal', bias_initializer='zeros'))

        model.summary()
        model.compile(optimizers.Adam(self.learning_rate), loss='mse')

        self.model = model

# Let's construct our agent !
custom_action_values = ActionValues(env.observation_space,
                                    env.action_space,
                                    freezed_steps=50,
                                    learning_rate=1e-2,
                                    learning_rate_decay=0)

agent = StandardAgent(observation_space=env.observation_space,
                      action_space=env.action_space,
                      action_values=custom_action_values,
                      forget_after_update=False,
                      exploration=.1,
                      exploration_decay=0,
                      memory_len=10000)

# Let him learn
pg = rl.Playground(env, agent)
pg.run(100,
       verbose=1,
       render=False,
       cycle_len=5,
       learn=True,
       # callbacks=[TensorboardCallback(log_dir='./logs_test/', run_name='test_model')]
       )
