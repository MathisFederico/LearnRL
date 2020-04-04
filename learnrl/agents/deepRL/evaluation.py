# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""
Evaluation methodes to modify the value fonctions from experiences
"""
from learnrl.core import Memory
from learnrl.agents.deepRL.control import Control
from learnrl.agents.deepRL.estimator import Estimator
import numpy as np

class Evaluation():

    """
    Basic evaluation object\n
    This method must be specified : learn(self, action_values, memory, learning_rate, action_visits=None).
    """

    def __init__(self, name=None, **kwargs):
        if name is None:
            raise ValueError("The Evaluation Object must have a name")
        self.name = name

    def eval(self, action_values:Estimator, action_visits:Estimator, memory:Memory, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.name


class MonteCarlo(Evaluation):

    """ MonteCarlo methods uses experimental mean to approximate theorical mean """

    def __init__(self, **kwargs):
        super().__init__(name="montecarlo", **kwargs)

    def eval(self, action_values:Estimator, action_visits:Estimator, memory:Memory, **kwargs):
        datas = memory.datas
        observation, action, reward, done, _, _ = [datas[key] for key in memory.MEMORY_KEYS]
        if np.any(done):
            total_return = np.sum(reward)
            memory.forget()
        return observation, action, total_return

class TemporalDifference(Evaluation):

    """ TemporalDifference uses previously computed action_values to approximate the expected return at each step """

    def __init__(self, **kwargs):
        super().__init__(name="tempdiff", **kwargs)
        self.target_control = kwargs.get('target_control')
        self.target_policy = self.target_control.get_policy if self.target_control else None
        self.online = kwargs.get('online', True)

    @staticmethod
    def _get_expected_futur_reward(action_values, action_visits, reward, done, policy, next_observation=None, target_policy=None):
        expected_futur_reward = reward.astype(np.float64)
        not_done = np.logical_not(done)
        if len(next_observation[not_done]) > 0:
            if target_policy is None:
                action_impact = policy(next_observation[not_done], action_values, action_visits) 
            else:
                action_impact = target_policy(next_observation[not_done], action_values, action_visits)
            expected_futur_reward[not_done] += np.sum(action_impact * action_values[next_observation[not_done], :], axis=-1)
        return expected_futur_reward

    def eval(self, action_values:Estimator, action_visits:Estimator, memory:Memory, control:Control, **kwargs):

        # Get specific parameters for TD
        policy = control.get_policy
        if policy is None:
            raise ValueError('You must specify a policy for TD evaluation')
        datas = memory.datas

        # If Online learning, learns every step
        observation, action, reward, done, next_observation, _ = [datas[key] for key in memory.MEMORY_KEYS]
        if np.any(done) or self.online:
            expected_futur_reward = self._get_expected_futur_reward(action_visits, action_values, reward, done,
                                                                    policy, next_observation, self.target_policy)
            memory.forget()
        
        return observation, action, expected_futur_reward

class QLearning(Evaluation):

    """
    QLearning is just TemporalDifference with Greedy target_control

    This object is just optimized for computation speed
    """

    def __init__(self, **kwargs):
        super().__init__(name="qlearning", **kwargs)
        self.online = kwargs.get('online', True)

    @staticmethod
    def _get_expected_futur_reward(action_values, action_visits, reward, done, policy, next_observation=None):
        expected_futur_reward = reward.astype(np.float64)
        not_done = np.logical_not(done)
        if len(next_observation[not_done]) > 0:
            expected_futur_reward[not_done] += np.amax(action_values(next_observation[not_done]), axis=-1)
        return expected_futur_reward
    
    def eval(self, action_values:Estimator, action_visits:Estimator, memory:Memory, control:Control, **kwargs):
        # Get specific parameters for TD
        policy = control.get_policy
        if policy is None:
            raise ValueError('You must specify a policy for TD evaluation')
        datas = memory.datas
        observation, action, reward, done, next_observation, _ = [datas[key] for key in memory.MEMORY_KEYS]
        if np.any(done) or self.online: # If Online learning, learns every step
            expected_futur_reward = self._get_expected_futur_reward(action_values, action_visits, reward, done,
                                                                    policy, next_observation)
            memory.forget()
        
        return observation, action, expected_futur_reward

