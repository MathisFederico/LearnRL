# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""
Evaluation methodes to modify the value functions from experiences
"""
from learnrl.memory import Memory
from learnrl.control import Control
from learnrl.estimators import Estimator

import numpy as np
from copy import copy
from itertools import cycle

class Evaluation():

    """ Evaluation base object
    
    The method eval must be specified.
    
    Example
    -------

    >>> from learnrl.evaluation import Evaluation
    ...
    ... class MyEvaluation(Evaluation):
    ... 
    ...     ''' Description '''
    ...
    ...     def __init__(self, **kwargs):
    ...         super().__init__(name="my_evaluation_name", **kwargs)
    ...
    ...     def eval(self, reward, done, next_observation, action_values:Estimator, action_visits:Estimator, control:Control):
    ...         ...
    ...         ...
    ...         return expected_returns
    
    Parameters
    ----------
        name: :class:`str`
            Name of the evaluation.
        discount: :class:`float` (default is .999)
           Discount factor.
           
    Attributes
    ----------
        All args becomes attributes.
    """

    def __init__(self, name=None, discount=.999, **kwargs):
        if name is None:
            raise ValueError("The Evaluation object must have a name")
        self.name = name
        self.discount = discount

    def eval(self, reward, done, next_observation, action_values:Estimator, action_visits:Estimator, control:Control):
        """ Evaluate the expected futur rewards
        
        Arguments
        ---------
            reward: float
                The real reward of the last step
            done: bool
                True if the environment has ended and previous step was the last.
            next_observation: np.ndarray
                The observation made after the step, used to predict what will happen next.
            action_values: :class:`~learnrl.estimator.Estimator`
                The action_values approximated by the agent.
            action_visits: :class:`~learnrl.estimators.Estimator`
                The action_visits approximated by the agent.
            control: :class:`~learnrl.control.Control`
                The control object used to predict agent behavior,
                can be the same (on-policy) or another (off-policy) that the agent is using.
        
        Return
        ------
            expected_return: np.ndarray
                The batch of expected returns (some of rewards) at the end of the episode for each experience.

        """
        raise NotImplementedError

    def _get_evaluation(self, reward, done, next_observation, action_values:Estimator, action_visits:Estimator, target_control:Control):
        expected_return = self.eval(reward, done, next_observation, action_values, action_visits, target_control)
        return expected_return

    def __str__(self):
        return self.name


class MonteCarlo(Evaluation):

    """ MonteCarlo methods uses experimental mean to approximate theorical mean. """

    def __init__(self, **kwargs):
        super().__init__(name="montecarlo", **kwargs)

    def eval(self, reward, done, next_observation, action_values:Estimator, action_visits:Estimator, control:Control):
        total_return = np.sum(reward) * np.ones_like(reward)
        return total_return

class TemporalDifference(Evaluation):

    """ TemporalDifference uses previously computed action_values to approximate the expected return at each step. """

    def __init__(self, **kwargs):
        super().__init__(name="tempdiff", **kwargs)

    def eval(self, reward, done, next_observation, action_values:Estimator, action_visits:Estimator, control:Control):
        expected_futur_reward = reward.astype(np.float64)
        not_done = np.logical_not(done)

        if len(next_observation[not_done]) > 0:
            policy = control._get_policy
            action_impact = policy(next_observation[not_done], action_values, action_visits)
            expected_futur_reward[not_done] += self.discount * np.sum(action_impact * action_values(next_observation[not_done]), axis=-1)

        return expected_futur_reward

class QLearning(Evaluation):

    """
    QLearning is just TemporalDifference with Greedy target_control.

    This object is just optimized for computation speed. """

    def __init__(self, **kwargs):
        super().__init__(name="qlearning", **kwargs)

    def eval(self, reward, done, next_observation, action_values:Estimator, action_visits:Estimator, control:Control):
        expected_futur_reward = reward.astype(np.float64)
        not_done = np.logical_not(done)

        if len(next_observation[not_done]) > 0:
            expected_futur_reward[not_done] += self.discount * np.amax(action_values(next_observation[not_done]), axis=-1)
        
        return expected_futur_reward

