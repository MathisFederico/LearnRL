"""
Evaluation methodes to modify the value fonctions from experiences
"""
from agents.agent import Memory
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

    def learn(self, action_visits, action_values, memory:Memory, **kwargs):
        raise NotImplementedError


class MonteCarlo(Evaluation):

    def __init__(self, initial_learning_rate=0.1):
        super().__init__(initial_learning_rate=initial_learning_rate, name="mc")

    def learn(self, action_visits, action_values, memory:Memory, **kwargs):
        datas = memory.datas

        if np.any(datas['done']):
            total_return = np.sum(datas['reward'])
            for state, action in zip(datas['state'], datas['action']):
                try:
                    action_visits[(state, action)] += 1
                    delta = total_return - action_values[(state, action)]
                    action_values[(state, action)] += self.learning_rate * delta
                
                # If unknown (state, action) couple
                except KeyError:
                    action_visits[(state, action)] = 1
                    action_values[(state, action)] = self.learning_rate * total_return
            
            memory.forget()


class TemporalDifference(Evaluation):

    def __init__(self, initial_learning_rate=0.1, target_policy=None):
        super().__init__(initial_learning_rate=initial_learning_rate, name="td")
        self.target_policy = None

    @staticmethod
    def get_next_legal_actions(next_state, memory):
        if next_state in memory.legal_actions:
            return memory.legal_actions[next_state]
        else:
            raise KeyError(f'Couldn\'t find state {next_state} legal_action\'s in agent memory')

    def get_expected_futur_reward(self, action_values, action, reward, done, policy, next_state=None, next_legal_actions=None, target_policy=None):
        expected_futur_reward = reward
        if not done:
            for next_action in next_legal_actions:
                # On vs Off Policy
                if target_policy is not None:
                    action_impact = target_policy(next_state, next_legal_actions)[next_action]
                else:
                    action_impact = policy(next_state, next_legal_actions)[next_action]
                # Only add expected reward from actions we took at least once
                if (next_state, next_action) in action_values:
                    expected_futur_reward += action_impact*action_values[(next_state, next_action)]
       
        return expected_futur_reward
    
    def learn_trajectory(self, action_visits, action_values, state, action, expected_futur_reward):
        try:
            action_visits[(state, action)] += 1
            delta = expected_futur_reward - action_values[(state, action)]
            action_values[(state, action)] += self.learning_rate * delta
        # If unknown (state, action) couple
        except KeyError:
            action_visits[(state, action)] = 1
            action_values[(state, action)] = self.learning_rate * expected_futur_reward


    def learn(self, action_visits, action_values, memory:Memory, **kwargs):

        # Get specific parameters for TD
        policy = kwargs.get('policy', None)
        if policy is None:
            raise ValueError('You must specify a policy for TD evaluation')
        online = kwargs.get('online', True)
        datas = memory.datas

        # If Online learning, learns every step
        if online:
            state, action, reward, done, next_state, _ = [datas[key][-1] for key in memory.MEMORY_KEYS]
            if done:
                self.learn_trajectory(action_visits, action_values, state, action, reward)
            # If not done, we estimate futur rewards based on memory and value fonctions
            else:
                # Only learn if we know what is next_state
                if next_state in memory.legal_actions:
                    next_legal_actions = self.get_next_legal_actions(next_state, memory)
                    expected_futur_reward = self.get_expected_futur_reward(action_values, action, reward, done, policy,
                                                                        next_state, next_legal_actions, self.target_policy)
                    self.learn_trajectory(action_visits, action_values, state, action, expected_futur_reward)
            memory.forget()
        
        # If Offline learning, learns in a batch at the end of the episode
        # else:
        #     if np.any(datas['done']):
        #         for state, action, reward, done, next_state in zip(datas['state'], datas['action'], datas['reward'], datas['done'], datas['next_state']):
        #             expected_futur_reward = self.get_expected_futur_reward(action_values, action, reward, done, policy, next_state, self.target_policy)
        #             self.learn_trajectory(action_values, action_visits, state, action, expected_futur_reward)
        #         memory.forget()