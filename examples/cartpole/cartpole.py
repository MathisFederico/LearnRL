import gym
import numpy as np

import learnrl as rl
from learnrl.agents import StandardAgent

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import Sequential

# Load the environment
env = gym.make('CartPole-v1')

# Define the model for our action-value estimator
class MyEstimator(rl.estimators.KerasEstimator):
    def preprocess(self, observations, actions):
        return observations

    def build(self, **kwargs):
        self.model = models.Sequential()
        self.model.add(layers.Flatten(input_shape=self.observation_shape))
        self.model.add(layers.Dense(self.observation_size, activation='tanh'))
        self.model.add(layers.Dense(self.action_size, activation='linear'))

        self.model.compile(optimizers.Adam(learning_rate=self.learning_rate), loss='mse')

# Define a reward scaler to avoid huge reward (more difficult to predict)
class RewardScaler(rl.RewardHandler):
    def reward(self, observation, action, reward, done, info, next_observation):
        return reward / 500.0

reward_handler = RewardScaler();


# Let's construct our agent !
custom_action_values = MyEstimator(env.observation_space,
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
playground = rl.Playground(env, agent)

# Train for 500 episodes
playground.fit(500,
               verbose=1,
               reward_handler=reward_handler,
               callbacks=[]) # Add Tensorboard, wandb or your own callbacks here

# Show its performance for 5 episodes
playground.test(5, verbose=1)
