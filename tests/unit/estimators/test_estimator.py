# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
import sys
import importlib

import learnrl
from learnrl import Estimator
from gym.spaces import Discrete

class DummyEstimator(Estimator):

    def build(self, **kwargs):
        pass

def test_update_learning_rate():

    initial_learning_rate = 0.378
    learning_rate_decay = 1e-1

    observation_space = Discrete(4)
    action_space = Discrete(3)

    evaluation = DummyEstimator(observation_space, action_space, learning_rate=initial_learning_rate, learning_rate_decay=learning_rate_decay)
    if evaluation.learning_rate != initial_learning_rate:
        raise ValueError(f"Learning rate is {evaluation.learning_rate} instead of initial_learning rate {initial_learning_rate}")
    
    learning_rate = 1.0
    evaluation.update_learning_rate(learning_rate=learning_rate)
    if evaluation.learning_rate != learning_rate:
        raise ValueError(f"Learning rate is {evaluation.learning_rate} instead of fixed updated learning rate {learning_rate}")
    
    evaluation.update_learning_rate()
    expected_learning_rate = learning_rate*(1-learning_rate_decay)
    if evaluation.learning_rate != expected_learning_rate:
        raise ValueError(f"Learning rate is {evaluation.learning_rate} and did not decay to {expected_learning_rate}")

