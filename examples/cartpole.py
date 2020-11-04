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

# Define a reward scaler to avoid huge reward (more difficult to predict)
class RewardScaler(rl.RewardHandler):
    def reward(self, observation, action, reward, done, info, next_observation):
        return reward / 500.0

reward_handler = RewardScaler();

# Define the model for our Estimator
class ActionValues(KerasEstimator):
    def preprocess(self, observations, actions):
        return observations

    def build(self, **kwargs):
        print(self.observation_size)
        print(self.action_size)

        model = Sequential()
        model.add(Input(shape=(self.observation_size,)))
        model.add(layers.Dense(self.observation_size, activation='tanh', kernel_initializer='random_normal', bias_initializer='zeros'))
        model.add(layers.Dense(self.action_size, activation='linear', kernel_initializer='random_normal', bias_initializer='zeros'))

        model.summary()
        model.compile(optimizers.Adam(self.learning_rate, decay=1e-6), loss='mse')

        self.model = model

# Let's construct our agent !
custom_action_values = ActionValues(env.observation_space,
                                    env.action_space,
                                    freezed_steps=200,
                                    learning_rate=1e-3,
                                    batch_size=32,
                                    epochs_per_step=1,
                                    learning_rate_decay=0)

agent = StandardAgent(observation_space=env.observation_space,
                      action_space=env.action_space,
                      action_values=custom_action_values,
                      forget_after_update=False,
                      exploration=.3,
                      exploration_decay=2e-5,
                      memory_len=10000)

# Let him learn
pg = rl.Playground(env, agent)
pg.run(1000,
       verbose=1,
       render=False,
       cycle_len=25,
       learn=True,
       reward_handler=reward_handler,
       # callbacks=[TensorboardCallback(log_dir='./logs_test/', run_name='test_model')]
       )
