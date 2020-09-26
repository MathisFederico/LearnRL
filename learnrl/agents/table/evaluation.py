# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""
Evaluation methodes to modify the value fonctions from experiences
"""
from learnrl.core import Memory
from learnrl.agents.table.control import Control
import numpy as np

class Evaluation():

    """
    Basic evaluation object\n
    This method must be specified : learn(self, action_values, memory, learning_rate, action_visits=None).
    """

    def __init__(self, initial_learning_rate=0.1, name=None, **kwargs):
        self.learning_rate = initial_learning_rate
        
        if name is None:
            raise ValueError("The Evaluation Object must have a name")
        self.name = name
        self.decay = kwargs.get('learning_rate_decay', 1)

    def learn(self, action_visits, action_values, memory:Memory, **kwargs):
        raise NotImplementedError
    
    def update_learning_rate(self, learning_rate=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate *= self.decay

    def __str__(self):
        return self.name


class MonteCarlo(Evaluation):

    """ MonteCarlo methods uses experimental mean to approximate theorical mean """

    def __init__(self, initial_learning_rate=0.1, **kwargs):
        super().__init__(initial_learning_rate=initial_learning_rate, name="montecarlo", **kwargs)

    def learn(self, action_visits, action_values, memory:Memory, **kwargs):
        datas = memory.datas

        if np.any(datas['done']):
            total_return = np.sum(datas['reward'])

            action_visits[datas['observation'], datas['action']] += 1
            delta = total_return - action_values[datas['observation'], datas['action']]
            action_values[datas['observation'], datas['action']] += self.learning_rate * delta

            memory.forget()


class TemporalDifference(Evaluation):

    """ TemporalDifference uses previously computed action_values to approximate the expected return at each step """

    def __init__(self, initial_learning_rate=0.1, **kwargs):
        super().__init__(initial_learning_rate=initial_learning_rate, name="tempdiff", **kwargs)
        self.target_control = kwargs.get('target_control')
        self.target_policy = self.target_control.get_policy if self.target_control else None
        self.online = kwargs.get('online', True)

    @staticmethod
    def _get_expected_futur_reward(action_visits, action_values, action, reward, done, policy, next_observation=None, target_policy=None):
        expected_futur_reward = reward.astype(np.float64)

        not_done = np.logical_not(done)
        if len(next_observation[not_done]) > 0:
            if target_policy is None:
                action_impact = policy(next_observation[not_done], action_values, action_visits) 
            else:
                action_impact = target_policy(next_observation[not_done], action_values, action_visits)
            expected_futur_reward[not_done] += np.sum(action_impact * action_values[next_observation[not_done], :], axis=-1)
        
        return expected_futur_reward
    
    def _learn_trajectory(self, action_visits, action_values, observation, action, expected_futur_reward):
        action_visits[observation, action] += 1
        delta = expected_futur_reward - action_values[observation, action]
        action_values[observation, action] += self.learning_rate * delta

    def learn(self, action_visits, action_values, memory:Memory, control:Control, **kwargs):

        # Get specific parameters for TD
        policy = control.get_policy
        if policy is None:
            raise ValueError('You must specify a policy for TD evaluation')
        datas = memory.datas

        # If Online learning, learns every step
        observation, action, reward, done, next_observation, _ = [datas[key] for key in memory.MEMORY_KEYS]
        if np.any(done) or self.online:
            expected_futur_reward = self._get_expected_futur_reward(action_visits, action_values, action, reward, done,
                                                                    policy, next_observation, self.target_policy)
            self._learn_trajectory(action_visits, action_values, observation, action, expected_futur_reward)
            memory.forget()

class QLearning(Evaluation):

    """
    QLearning is just TemporalDifference with Greedy target_control

    This object is just optimized for computation speed
    """

    def __init__(self, initial_learning_rate=0.1, **kwargs):
        super().__init__(initial_learning_rate=initial_learning_rate, name="qlearning", **kwargs)
        self.online = kwargs.get('online', True)

    @staticmethod
    def _get_expected_futur_reward(action_visits, action_values, action, reward, done, policy, next_observation=None):
        expected_futur_reward = reward.astype(np.float64)
        not_done = np.logical_not(done)
        if len(next_observation[not_done]) > 0:
            expected_futur_reward[not_done] += np.amax(action_values[next_observation[not_done], :], axis=-1)
        return expected_futur_reward
    
    def _learn_trajectory(self, action_visits, action_values, observation, action, expected_futur_reward):
        action_visits[observation, action] += 1
        delta = expected_futur_reward - action_values[observation, action]
        action_values[observation, action] += self.learning_rate * delta
    
    def learn(self, action_visits, action_values, memory:Memory, control:Control, **kwargs):

        # Get specific parameters for TD
        policy = control.get_policy
        if policy is None:
            raise ValueError('You must specify a policy for TD evaluation')
        datas = memory.datas

        observation, action, reward, done, next_observation, _ = [datas[key] for key in memory.MEMORY_KEYS]
        if np.any(done) or self.online: # If Online learning, learns every step
            expected_futur_reward = self._get_expected_futur_reward(action_visits, action_values, action, reward, done,
                                                                    policy, next_observation)
            self._learn_trajectory(action_visits, action_values, observation, action, expected_futur_reward)
            memory.forget()

