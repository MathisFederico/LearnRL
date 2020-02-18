"""
Evaluation methodes to modify the value fonctions from experiences
"""

import numpy as np

class Evaluation():

    name = 'defaulteval'

    def learn(self, action_visits, action_values, memory, learning_rate):
        raise NotImplementedError


class MonteCarlo(Evaluation):

    name = 'mc'
    
    def learn(self, action_visits, action_values, memory, learning_rate):
        datas = memory.datas

        if np.any(datas['done']):

            total_return = np.sum(datas['reward'])

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