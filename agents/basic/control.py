"""
Control methodes to improve the policy based on value fonctions
"""

import numpy as np


class Control():

    name = 'defaultcontrol'
    
    def checkPolicy(self, policy):
        if policy is None:
            raise NotImplementedError(self.getPolicy)
        try: 
            assert np.all(policy >= 0), "Policy have probabilities < 0"
            assert np.sum(policy, axis=-1) == 1.0, "Policy probabilities does not sum to 1"
        except AssertionError as error:
            print(f"Policy is not valid : {error}")

    def getPolicy(self, **params):
        policy = None
        self.checkPolicy(policy)
        return policy

class Greedy(Control):

    name ='greedy'

    def getPolicy(self, **params):
        action_values = params.get('action_values')
        exploration_coef = params.get('exploration_coef')

        best_action_id = np.argmax(action_values)

        policy = np.ones(action_values.shape) * exploration_coef / action_values.shape[-1]
        policy[best_action_id] += 1 - exploration_coef

        self.checkPolicy(policy)
        return policy


class UCB(Control):

    name ='ucb'

    def getPolicy(self, **params):
        action_visits = params.get('action_visits')
        action_values = params.get('action_values')
        exploration_coef = params.get('exploration_coef')

        best_action_id = np.argmax(action_values + \
                              exploration_coef * np.power(np.log(1+np.sum(action_visits))/action_visits, 1/2))
        
        policy = np.zeros(action_values.shape)
        policy[best_action_id] = 1.0

        self.checkPolicy(policy)
        return policy