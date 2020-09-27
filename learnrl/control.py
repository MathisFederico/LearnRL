# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""
Control methodes to improve the policy based on value fonctions
"""
import numpy as np
from learnrl.estimators import Estimator

class Control():

    """ Control base object\n

    This method must be specified :
    policy(self, action_values, action_visits=None).

    Example
    -------

    >>> from learnrl.control import Control
    ...
    ... class MyControl(Control):
    ...
    ...     def __init__(self, exploration=0.1, **kwargs):
    ...         super().__init__(exploration=exploration, name="my_control_name", **kwargs)
    ...         self.need_action_visit = False/True
    ...
    ...     def policy(self, observations:np.ndarray, action_values:Estimator, action_visits:Estimator=None):
    ...         ...
    ...         ...
    ...         return policy

    Parameters
    ----------
        exploration: float
            The initial exploration constant
        name: str
            The name of the control (mandatory)
        exploration_decay: float
            The exploration_decay (default to 0)
        
    Attributes
    ----------
        exploration: float
            The exploration constant
        name: str
            The name of the control
        decay: float
            The exploration decay
        need_action_visit: bool
            True if the control needs action visit

    """

    def __init__(self, exploration=0, name=None, **kwargs):
        self.exploration = exploration
        if name is None:
            raise ValueError("The Control object must have a name")
        self.name = name
        self.decay = kwargs.get('exploration_decay', 0)
        self.need_action_visit = False

    def policy(self, Q, N=None):
        """ Return the policy of the agent given an observation
        
        Arguments
        ---------
            Q: numpy.ndarray
                The estimator of Q(s,a), the expected futur reward if we do action 'a' in state 's'.
                Shape is (sample_size, action_size).

            N: numpy.ndarray, optional
                The estimator of N(s,a), the number of times we did action 'a' in state 's'.
                Shape is (sample_size, action_size).
        
        Return
        ------
            policy: numpy.ndarray
                The probabilities of choosing every actions.

        """
        raise NotImplementedError

    def _get_policy(self, observations:np.ndarray, action_values:Estimator, action_visits:Estimator=None, greedy=False):
        if action_visits is None and self.need_action_visit:
            raise ValueError(f"action_visits must be specified for {self.name} control")

        Q = action_values(observations)
        N = action_visits(observations) if self.need_action_visit else None
        policies = self.policy(Q, N)
        self._check_policy(policies)
        return policies
    
    def _check_policy(self, policies):
        if np.any(policies < 0) or np.any(policies > 1):
            raise ValueError("Policy are not in [0, 1]")
        prob_sums = np.sum(policies, axis=1)
        invalid_policy = np.abs(np.sum(prob_sums - 1)) > 1e-6
        if np.any(invalid_policy):
            raise ValueError(f"Policy {policies[invalid_policy]} probabilities sum to {prob_sums[invalid_policy]} and not 1")

    def update_exploration(self, exploration=None):
        """ Update the exploration constant

        By default, uses exploration_decay (or self.decay) to update exploration 
        with formula :math:`exploration *= (1 - decay)`.

        If this method is called with a float argument, exploration will update to that float.
        
        Arguments
        ---------
            exploration: float, optional
                The fixed exploration constant that we set to.
        
        """
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

    """ Random control to use for tests """

    def __init__(self, exploration=1, **kwargs):
        super().__init__(exploration=exploration, name="random", **kwargs)

    def policy(self, Q:np.ndarray, N:np.ndarray=None):
        sample_size, action_size = Q.shape
        p = np.zeros(Q.shape)
        action = np.random.choice(action_size, size=(sample_size,), replace=True)
        p[:, action] = 1
        return p

class Greedy(Control):

    """ Greedy control
    
    Takes the action maximizing action_value with probability (1-exploration).
    Else take a uniformly random action.

    """

    def __init__(self, exploration=0.1, **kwargs):
        super().__init__(exploration=exploration, name="greedy", **kwargs)

    def policy(self, Q:np.ndarray, N:np.ndarray=None):
        if self.exploration < 0 or self.exploration > 1:
            raise ValueError(f"Exploration should be in [0, 1] for greedy control but was {self.exploration}")
            
        best_action_id = np.argmax(Q, axis=1)
        _, action_size = Q.shape
        policy = np.ones_like(Q) * self.exploration / action_size
        policy[np.arange(len(best_action_id)), best_action_id] += 1 - self.exploration
        return policy

class Ucb(Control):

    """ Upper Confidence Bound control

    """

    def __init__(self, exploration=1, **kwargs):
        super().__init__(exploration=exploration, name="ucb", **kwargs)
        self.need_action_visit = True

    def policy(self, Q:np.ndarray, N:np.ndarray=None):
        best_action_id = np.argmax(Q + self.exploration * np.sqrt( np.log(1 + np.sum(N, axis=1)) / (1.0 + N)), axis=1)
        policy = np.zeros_like(Q, dtype=np.uint8)
        policy[np.arange(len(best_action_id)), best_action_id] = 1
        return policy


class Puct(Control):

    """ Puct control

    """

    def __init__(self, exploration=1, **kwargs):
        super().__init__(exploration=exploration, name="puct", **kwargs)
        self.need_action_visit = True

    def policy(self, Q:np.ndarray, N:np.ndarray=None):  
        best_action_id = np.argmax(Q + self.exploration * np.sqrt( np.sum(N, axis=1) / (1.0 + N)), axis=1)
        policy = np.zeros_like(Q, dtype=np.uint8)
        policy[np.arange(len(best_action_id)), best_action_id] = 1
        return policy
