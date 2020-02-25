"""
Evaluation methodes to modify the value fonctions from experiences
"""

import numpy as np

class Evaluation():

    """
    Basic evaluation object\n
    This method must be specified : learn(self, action_values, memory, learning_rate, action_visits=None).
    """

    def __init__(self, initial_learning_rate=0.1, name=None):
        self.learning_rate = initial_learning_rate

        if name is None:
            raise ValueError("The Control Object must have a name")

        self.name = name
    
    def update_learning_rate(self):
        pass

    def learn(self, action_visits, action_values, memory):
        raise NotImplementedError


class MonteCarlo(Evaluation):

    def __init__(self, initial_learning_rate=0.1):
        super().__init__(initial_learning_rate=initial_learning_rate, name="mc")

    def learn(self, action_visits, action_values, memory):
        datas = memory.datas

        if np.any(datas['done']):
            total_return = np.sum(datas['reward'])
            for state_id, action_id in zip(datas['state'], datas['action']):
                try:
                    action_visits[(state_id, action_id)] += 1
                    delta = total_return - action_values[(state_id, action_id)]
                    action_values[(state_id, action_id)] += self.learning_rate * delta
                
                # If unknown (state, action) couple
                except KeyError:
                    action_visits[(state_id, action_id)] = 1
                    action_values[(state_id, action_id)] = self.learning_rate * total_return
            
            memory.forget()


class TemporalDifference(Evaluation):

    def __init__(self, initial_learning_rate=0.1):
        super().__init__(initial_learning_rate=initial_learning_rate, name="td")

    def learn(self, action_visits, action_values, memory, target_policy=None, online=False):

        def know_next_state(action_visits, next_state):
            return np.any(np.array([*action_visits]))
            
        def get_expected_futur_reward(action_values, action, reward, done, next_state=None, next_legal_actions=None, target_policy=None):
            expected_futur_reward = reward
            if not done:
                for next_action in next_legal_actions:
                    expected_futur_reward += target_policy(next_state, next_legal_actions)[next_action]*action_values[(next_state, next_action)]
            return expected_futur_reward

        def td_learn(action_visits, action_values, state, action, expected_futur_reward):
            try:
                action_visits[(state, action)] += 1
                delta = expected_futur_reward - action_values[(state, action)]
                action_values[(state, action)] += self.learning_rate * delta
            # If unknown (state, action) couple
            except KeyError:
                action_visits[(state, action)] = 1
                action_values[(state, action)] = self.learning_rate * delta

        datas = memory.datas
        # If Online learning, learns every step
        if online:
            last_datas = {key:datas[key][-1] for key in datas}
            expected_futur_reward = get_expected_futur_reward(action_values, last_datas['action'], last_datas['reward'], last_datas['done'],
                                                                             last_datas['next_state'], target_policy)
            td_learn(action_visits, action_values, last_datas['state'], last_datas['action'], expected_futur_reward)
            memory.forget()
        
        # If Offline learning, only learns at the end of an episode
        else:
            if np.any(datas['done']):
                for state, action, reward, done, next_state in zip(datas['state'], datas['action'], datas['reward'], datas['done'], datas['next_state']):
                    expected_futur_reward = get_expected_futur_reward(action_values, action, reward, done, next_state, target_policy)
                    td_learn(action_values, action_visits, state, action, expected_futur_reward)
                memory.forget()