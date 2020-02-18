"""
Control methodes to improve the policy based on value fonctions
"""

import numpy as np

class Control():

    """ Base control object
    The getPolicy(self, **params) method must be specified
        **params is a dictionary containing value fonctions and visit counts
        typical uses are :
            action_visits = params.get('action_visits') to get Q(s,a)
            action_values = params.get('action_values') to get N(s,a)
    It is adviced to call self.checkPolicy(policy) before returning it
    """
    
    name = 'defaultcontrol'
    
    def checkPolicy(self, policy):
        try: 
            assert np.all(policy >= 0), "Policy have probabilities < 0"
            prob_sum = np.sum(policy, axis=-1)
            assert np.abs(prob_sum-1) <= 1e-9, "Policy probabilities sum to {} and not 1".format(prob_sum)
        except AssertionError as error:
            print("Policy is not valid :\n\t{}".format(error))

    def getPolicy(self, **params):
        raise NotImplementedError

class Greedy(Control):

    name ='greedy'

    def getPolicy(self, **params):
        action_values = params.get('action_values')
        exploration = params.get('exploration', 0)

        best_action_id = np.argmax(action_values)

        policy = np.ones(action_values.shape) * exploration / action_values.shape[-1]
        policy[best_action_id] += 1 - exploration

        self.checkPolicy(policy)
        return policy


class UCB(Control):

    name ='ucb'

    def getPolicy(self, **params):
        action_visits = params.get('action_visits')
        action_values = params.get('action_values')
        exploration = params.get('exploration', 0)

        best_action_id = np.argmax(action_values + \
                              exploration * np.sqrt(np.log(1+np.sum(action_visits))/(1.0+action_visits)))
        policy = np.zeros(action_values.shape)
        policy[best_action_id] = 1.0

        self.checkPolicy(policy)
        return policy


class Puct(Control):

    def getPolicy(self, **params):
        action_visits = params.get('action_visits')
        action_values = params.get('action_values')
        exploration = params.get('exploration', 0)

        best_action_id = np.argmax(action_values + \
                              exploration * np.sqrt(np.sum(action_visits)/(1.0+action_visits)))
        
        policy = np.zeros(action_values.shape)
        policy[best_action_id] = 1.0

        self.checkPolicy(policy)
        return policy