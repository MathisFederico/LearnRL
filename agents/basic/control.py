"""
Control methodes to improve the policy based on value fonctions
"""

import numpy as np

class Control():

    """ 
    Base control object\n
    This method must be specified :
    getPolicy(self, **params).

    You can get agent knowledge with **param.
    Exploration constants are an argument of the control object.
    You can see the Greedy object below for exemple.

    It is adviced to call self.checkPolicy(policy) before returning it.
    """

    def __init__(self, initial_exploration=0, name=None):
        self.exploration = initial_exploration

        if name is None:
            raise ValueError("The Control Object must have a name")

        self.name = name

    def updateExploration(self, exploration=None):
        if exploration is not None:
            self.exploration = exploration

    def checkPolicy(self, policy):
        try: 
            assert np.all(policy >= 0), "Policy have probabilities < 0"
            prob_sums = np.sum(policy, axis=-1)
            valid_policy = np.abs(np.sum(prob_sums - 1)) <= 1e-8
            assert np.all(valid_policy), f"Policy probabilities sum to {prob_sums[not valid_policy]} and not 1"
        except AssertionError as error:
            print(f"Policy is not valid : {error}")

    def getPolicy(self, action_values, action_visits=None):
        raise NotImplementedError

class Greedy(Control):

    def __init__(self, initial_exploration=0, decay=1):
        super().__init__(initial_exploration=initial_exploration, name="greedy")
        self.decay = decay

    def updateExploration(self, exploration=None):
        self.exploration *= self.decay

    def getPolicy(self, action_values, action_visits=None):
        best_action_id = np.argmax(action_values, axis=-1)

        policy = np.ones(action_values.shape) * self.exploration / action_values.shape[-1]
        if policy.ndim > 1:
            i = np.arange(policy.shape[0])
            policy[i, best_action_id] += 1 - self.exploration
        else:
            policy[best_action_id] += 1 - self.exploration

        self.checkPolicy(policy)
        return policy


class UCB(Control):

    def __init__(self, initial_exploration=1):
        super().__init__(initial_exploration=initial_exploration, name="ucb")

    def getPolicy(self, action_values, action_visits=None):
        if action_visits is None:
            raise ValueError("action_visits must be specified")

        best_action_id = np.argmax(action_values + \
                              self.exploration * np.sqrt(np.log(1+np.sum(action_visits, axis=-1))/(1.0+action_visits)), axis=-1)

        policy = np.zeros(action_values.shape)
        policy[best_action_id] = 1.0

        self.checkPolicy(policy)
        return policy


class Puct(Control):

    def __init__(self, initial_exploration=1):
        super().__init__(initial_exploration=initial_exploration, name="puct")

    def getPolicy(self, action_values, action_visits=None):
        if action_visits is None:
            raise ValueError("action_visits must be specified")
        
        best_action_id = np.argmax(action_values + \
                              self.exploration * np.sqrt(np.sum(action_visits, axis=-1)/(1.0+action_visits)), axis=-1)
        
        policy = np.zeros(action_values.shape)
        policy[best_action_id] = 1.0

        self.checkPolicy(policy)
        return policy
