# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""
Control methodes to improve the policy based on value fonctions
"""
import numpy as np
from learnrl.agents.deepRL.estimator import Estimator

class Control():

    """ 
    Base control object\n
    This method must be specified :
    policy(self, action_values, action_visits=None).
    
    Exploration constants are an argument of the control object.
    You can see the Greedy(Control) object for exemple.
    """

    def __init__(self, exploration=0, name=None, **kwargs):
        self.exploration = exploration
        if name is None:
            raise ValueError("The Control Object must have a name")
        self.name = name
        self.decay = kwargs.get('exploration_decay', 1)

    def policy(self, observation, action_values:Estimator, action_visits:Estimator=None):
        raise NotImplementedError

    def get_policy(self, observations, action_values:Estimator, action_visits:Estimator=None):
        if len(observations) == 1:
            policy = self.policy(observations[0], action_values, action_visits)
            self.check_policy(policy)
            return policy
        else:
            policies = []
            for observation in observations:
                policy = self.policy(observation, action_values, action_visits)
                self.check_policy(policy)
                policies.append(policy)
            return np.array(policies)
    
    def check_policy(self, policy):
        if not np.all(policy >= 0):
            raise ValueError("Policy have probabilities < 0")
        prob_sums = np.sum(policy, axis=-1)
        valid_policy = np.abs(np.sum(prob_sums - 1)) <= 1e-8
        if not np.all(valid_policy):
            raise ValueError(f"Policy {policy} probabilities sum to {prob_sums[np.logical_not(valid_policy)]} and not 1")

    def update_exploration(self, exploration=None):
        if exploration is not None:
            self.exploration = exploration
        else:
            self.exploration *= self.decay

    def __str__(self):
        return f'{self.name}_exp({self.exploration:.2f}_decay({self.decay}))'
    
    def __eq__(self, other):
        same_name = self.name == other.name
        same_exploration = self.exploration == other.exploration
        same_decay = self.decay == other.decay
        return same_name and same_exploration and same_decay

class Greedy(Control):

    def __init__(self, exploration=0.1, **kwargs):
        super().__init__(exploration=exploration, name="greedy", **kwargs)

    def policy(self, observation, action_values:Estimator, action_visits:Estimator=None):
        Q = action_values(observation)
        best_action_id = np.argmax(Q)
        policy = np.ones_like(Q) * self.exploration / len(Q)
        policy[best_action_id] += 1 - self.exploration
        return policy


class UCB(Control):

    def __init__(self, exploration=1, **kwargs):
        super().__init__(exploration=exploration, name="ucb", **kwargs)

    def policy(self, observation, action_values:Estimator, action_visits:Estimator=None):
        if action_visits is None:
            raise ValueError("action_visits must be specified for UCB")

        N = action_visits(observation)
        Q = action_values(observation)
        best_action_id = np.argmax(Q + self.exploration * np.sqrt( np.log(1 + np.sum(N)) / (1.0 + N)))

        policy = np.zeros_like(Q)
        policy[best_action_id] = 1.0
        return policy


class Puct(Control):

    def __init__(self, exploration=1, **kwargs):
        super().__init__(exploration=exploration, name="puct", **kwargs)
        self.decay = kwargs.get('decay', 1)

    def policy(self, observation, action_values:Estimator, action_visits:Estimator=None):
        if action_visits is None:
            raise ValueError("action_visits must be specified for Puct")
        
        N = action_visits(observation)
        Q = action_values(observation)
        best_action_id = np.argmax(Q + self.exploration * np.sqrt( np.sum(N) / (1.0 + N)))
        
        policy = np.zeros_like(Q)
        policy[best_action_id] = 1.0
        return policy
