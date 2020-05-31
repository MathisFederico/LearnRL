# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""
Evaluation methodes to modify the value fonctions from experiences
"""
from learnrl.core import Memory
from learnrl.agent_parts.control import Control
from learnrl.agent_parts.estimator import Estimator

import numpy as np
from copy import copy
from itertools import cycle

class Evaluation():

    """ Basic evaluation object

    This method must be specified : eval(self, reward, done, next_observation, action_values:Estimator, action_visits:Estimator, control:Control)

    """

    def __init__(self, name=None, **kwargs):
        if name is None:
            raise ValueError("The Evaluation object must have a name")
        self.name = name

    def eval(self, reward, done, next_observation, action_values:Estimator, action_visits:Estimator, control:Control):
        raise NotImplementedError

    def get_evaluation(self, reward, done, next_observation, action_values:Estimator, action_visits:Estimator, target_control:Control):
        expected_return = self.eval(reward, done, next_observation, action_values, action_visits, target_control)
        return expected_return

    def __str__(self):
        return self.name


class MonteCarlo(Evaluation):

    """ MonteCarlo methods uses experimental mean to approximate theorical mean """

    def __init__(self, **kwargs):
        super().__init__(name="montecarlo", **kwargs)

    def eval(self, reward, done, next_observation, action_values:Estimator, action_visits:Estimator, control:Control):
        total_return = np.sum(reward) * np.ones_like(reward)
        return total_return

class TemporalDifference(Evaluation):

    """ TemporalDifference uses previously computed action_values to approximate the expected return at each step """

    def __init__(self, **kwargs):
        super().__init__(name="tempdiff", **kwargs)

    def eval(self, reward, done, next_observation, action_values:Estimator, action_visits:Estimator, control:Control):
        expected_futur_reward = reward.astype(np.float64)
        not_done = np.logical_not(done)

        if len(next_observation[not_done]) > 0:
            policy = control.get_policy
            action_impact = policy(next_observation[not_done], action_values, action_visits)
            expected_futur_reward[not_done] += np.sum(action_impact * action_values(next_observation[not_done]), axis=-1)

        return expected_futur_reward

class QLearning(Evaluation):

    """
    QLearning is just TemporalDifference with Greedy target_control

    This object is just optimized for computation speed
    """

    def __init__(self, **kwargs):
        super().__init__(name="qlearning", **kwargs)

    def eval(self, reward, done, next_observation, action_values:Estimator, action_visits:Estimator, control:Control):
        expected_futur_reward = reward.astype(np.float64)
        not_done = np.logical_not(done)

        if len(next_observation[not_done]) > 0:
            expected_futur_reward[not_done] += np.amax(action_values(next_observation[not_done]), axis=-1)
        
        return expected_futur_reward

