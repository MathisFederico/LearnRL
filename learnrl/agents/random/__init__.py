# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from learnrl import Agent
from learnrl.control import Random

class RandomAgent(Agent):

    def __init__(self, observation_space, action_space, control=None, evaluation=None, action_values=None, action_visits=None, **kwargs):
        super().__init__()
        self.name = "random"
        self.observation_space = observation_space
        self.action_space = action_space
        self.control = Random()
    
    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        pass

    def act(self, observation, greedy=False):
        return self.action_space.sample()
    
    def learn(self):
        pass

