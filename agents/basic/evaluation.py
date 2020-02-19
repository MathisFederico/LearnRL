"""
Evaluation methodes to modify the value fonctions from experiences
"""

import numpy as np

class Evaluation(object):

    """
    Basic evaluation object\n
    This method must be specified : learn(self, action_values, memory, learning_rate, action_visits=None).
    """

    name = 'defaulteval'

    def __init__(self, initial_learning_rate=0.1, name=None):
        self.learning_rate = initial_learning_rate

        if name is None:
            raise ValueError("The Control Object must have a name")

        self.name = name
    
    def update_learning_rate(self):
        pass

    def in_dict(self, dictionary, state, action):
        """Check efficiently if (state, action) is in dict"""
        keys = np.concatenate((state, action), axis=-1)
        res = np.isin(keys, dictionary)
        print(res)
        return res

    def learn(self, action_visits, action_values, memory, learning_rate):
        raise NotImplementedError


class MonteCarlo(Evaluation):

    def __init__(self, initial_learning_rate=0.1):
        super(MonteCarlo, self).__init__(initial_learning_rate=initial_learning_rate, name="mc")

    def learn(self, action_values, memory, learning_rate, action_visits=None):
        datas = memory.datas

        if np.any(datas['done']):

            total_return = np.sum(datas['reward'])

            # knowned_datas = np.map(datas['state'], datas['action'])

            # Have to be optimized
            for state_id, action in zip(datas['state'], datas['action']):
                try:
                    # Modify the action_visits N(s, a)
                    action_visits[(state_id, action)] += 1
                    # Modify the action_values Q(s, a)
                    delta = total_return - action_values[(state_id, action)]
                    action_values[(state_id, action)] += learning_rate * delta
                
                # If unknown (state, action) couple
                except KeyError:
                    # Define the action_visits N(s, a)
                    action_visits[(state_id, action)] = 1
                    # Define the action_values Q(s, a)
                    action_values[(state_id, action)] = learning_rate * total_return
            
            memory.forget()



class TemporalDifference(Evaluation):

    name = 'td'