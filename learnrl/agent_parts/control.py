# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""
Control methodes to improve the policy based on value fonctions
"""
import numpy as np
from learnrl.agent_parts.estimator import Estimator

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
            raise ValueError("The Control object must have a name")
        self.name = name
        self.decay = kwargs.get('exploration_decay', 0)
        self.need_action_visit = False

    def policy(self, observation:np.ndarray, action_values:Estimator, action_visits:Estimator=None):
        raise NotImplementedError

    def get_policy(self, observations:np.ndarray, action_values:Estimator, action_visits:Estimator=None, greedy=False):
        if action_visits is None and self.need_action_visit:
            raise ValueError(f"action_visits must be specified for {self.name} control")

        policy = self.policy if not greedy else Greedy(0).policy
        policies = np.empty((len(observations), action_values.action_size))
        for i, observation in enumerate(observations):
            policies[i] = policy(observation, action_values, action_visits)
        policies = np.stack(policies)

        self.check_policy(policies)
        return policies
    
    def check_policy(self, policies):
        if np.any(policies < 0) or np.any(policies > 1):
            raise ValueError("Policy are not in [0, 1]")
        prob_sums = np.sum(policies, axis=1)
        invalid_policy = np.abs(np.sum(prob_sums - 1)) > 1e-6
        if np.any(invalid_policy):
            raise ValueError(f"Policy {policies[invalid_policy]} probabilities sum to {prob_sums[invalid_policy]} and not 1")

    def update_exploration(self, exploration=None):
        if exploration is not None:
            self.exploration = exploration
        else:
            self.exploration *= (1 - self.decay)

    def __str__(self):
        return f'{self.name}_exp({self.exploration:.2f}_decay({self.decay}))'
    
    def __eq__(self, other):
        same_name = self.name == other.name
        same_exploration = self.exploration == other.exploration
        same_decay = self.decay == other.decay
        return same_name and same_exploration and same_decay


class Random(Control):

    def __init__(self, exploration=1, **kwargs):
        super().__init__(exploration=exploration, name="random", **kwargs)

    def policy(self, observations:np.ndarray, action_values:Estimator, action_visits:Estimator=None):
        action_size = action_values.action_size
        p = np.zeros((len(observations), action_size))
        action = np.random.choice(action_size, size=(len(observations),), replace=True)
        p[:, action] = 1
        return p

class Greedy(Control):

    def __init__(self, exploration=0.1, **kwargs):
        super().__init__(exploration=exploration, name="greedy", **kwargs)

    def policy(self, observations:np.ndarray, action_values:Estimator, action_visits:Estimator=None):
        if self.exploration < 0 or self.exploration > 1:
            raise ValueError(f"Exploration should be in [0, 1] for greedy control but was {self.exploration}")
        Q = action_values(observations)
        best_action_id = np.argmax(Q)
        policy = np.ones_like(Q) * self.exploration / Q.size
        policy[best_action_id] += 1 - self.exploration
        return policy

class Ucb(Control):

    def __init__(self, exploration=1, **kwargs):
        super().__init__(exploration=exploration, name="ucb", **kwargs)
        self.need_action_visit = True

    def policy(self, observations:np.ndarray, action_values:Estimator, action_visits:Estimator):
        N = action_visits(observations)
        Q = action_values(observations)
        best_action_id = np.argmax(Q + self.exploration * np.sqrt( np.log(1 + np.sum(N)) / (1.0 + N)))

        policy = np.zeros_like(Q)
        policy[best_action_id] = 1.0
        return policy


class Puct(Control):

    def __init__(self, exploration=1, **kwargs):
        super().__init__(exploration=exploration, name="puct", **kwargs)
        self.need_action_visit = True

    def policy(self, observations, action_values:Estimator, action_visits:Estimator):       
        N = action_visits(observations)
        Q = action_values(observations)
        best_action_id = np.argmax(Q + self.exploration * np.sqrt( np.sum(N) / (1.0 + N)))
        
        policy = np.zeros_like(Q)
        policy[best_action_id] = 1.0
        return policy