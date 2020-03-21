"""
Control methodes to improve the policy based on value fonctions
"""
import numpy as np


class Control():

    """ 
    Base control object\n
    This method must be specified :
    policy(self, action_values, action_visits=None).
    
    Exploration constants are an argument of the control object.
    You can see the Greedy(Control) object for exemple.
    """

    def __init__(self, action_size, initial_exploration=0, name=None, **kwargs):
        self.exploration = initial_exploration
        self.action_size = action_size
        if name is None:
            raise ValueError("The Control Object must have a name")
        self.name = name
        self.decay = kwargs.get('exploration_decay', 1)

    def policy(self, state, action_values, action_visits=None):
        raise NotImplementedError

    def get_policy(self, state, action_values, action_visits=None):

        if type(state) != np.ndarray or state.ndim < 1:
            policy = self.policy(state, action_values, action_visits)
            self.check_policy(policy)
            return policy

        if state.ndim == 1:
            state = state[:, np.newaxis]
        policies = np.apply_along_axis(self.policy, axis=1, arr=state, action_values=action_values, action_visits=action_visits)
        self.check_policy(policies)
        return policies
    
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

    def __init__(self, action_size, initial_exploration=0.1, **kwargs):
        super().__init__(action_size, initial_exploration=initial_exploration, name="greedy", **kwargs)

    def policy(self, state, action_values, action_visits=None):
        best_action_id = np.argmax(action_values[state])
        policy = np.ones(self.action_size) * self.exploration / self.action_size
        policy[best_action_id] += 1 - self.exploration
        return policy


class UCB(Control):

    def __init__(self, action_size, initial_exploration=1, **kwargs):
        super().__init__(action_size, initial_exploration=initial_exploration, name="ucb", **kwargs)

    def policy(self, state, action_values, action_visits=None):
        if action_visits is None:
            raise ValueError("action_visits must be specified for UCB")

        N = action_visits[state]
        Q = action_values[state]
        best_action_id = np.argmax(Q + self.exploration * np.sqrt( np.log(1 + np.sum(N)) / (1.0 + N)))

        policy = np.zeros(self.action_size)
        policy[best_action_id] = 1.0
        return policy


class Puct(Control):

    def __init__(self, action_size, initial_exploration=1, **kwargs):
        super().__init__(action_size, initial_exploration=initial_exploration, name="puct", **kwargs)
        self.decay = kwargs.get('decay', 1)

    def policy(self, state, action_values, action_visits=None):
        if action_visits is None:
            raise ValueError("action_visits must be specified for Puct")
        
        N = action_visits[state]
        Q = action_values[state]
        best_action_id = np.argmax(Q + self.exploration * np.sqrt( np.sum(N) / (1.0 + N)))
        
        policy = np.zeros(self.action_size)
        policy[best_action_id] = 1.0
        return policy
