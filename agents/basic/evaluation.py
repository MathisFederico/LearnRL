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

            action_visits[datas['state'], datas['action']] += 1
            delta = total_return - action_values[datas['state'], datas['action']]
            action_values[datas['state'], datas['action']] += self.learning_rate * delta

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

    def get_expected_futur_reward(self, action_values, action, reward, done, policy, next_state=None, target_policy=None):
        expected_futur_reward = reward.astype(np.float64)
        not_done = np.logical_not(done)
        action_impact = policy(next_state[not_done]) if target_policy is None else target_policy(next_state[not_done])
        expected_futur_reward[not_done] += np.sum(action_impact * action_values[next_state[not_done], :], axis=-1)
        return expected_futur_reward
    
    def learn_trajectory(self, action_visits, action_values, state, action, expected_futur_reward):
        action_visits[state, action] += 1
        delta = expected_futur_reward - action_values[state, action]
        action_values[state, action] += self.learning_rate * delta


    def learn(self, action_visits, action_values, memory:Memory, **kwargs):

        # Get specific parameters for TD
        policy = kwargs.get('policy', None)
        if policy is None:
            raise ValueError('You must specify a policy for TD evaluation')
        online = kwargs.get('online', True)
        datas = memory.datas

        # If Online learning, learns every step
        state, action, reward, done, next_state, _ = [datas[key] for key in memory.MEMORY_KEYS]
        if online:
            if np.any(done):
                self.learn_trajectory(action_visits, action_values, state, action, reward)
            # If not done, we estimate futur rewards based on memory and value fonctions
            else:
                expected_futur_reward = self.get_expected_futur_reward(action_values, action, reward, done, policy,
                                                                        next_state, self.target_policy)
                self.learn_trajectory(action_visits, action_values, state, action, expected_futur_reward)
            memory.forget()
        
        # If Offline learning, learns in a batch at the end of the episode
        else:
            if np.any(done):
                expected_futur_reward = self.get_expected_futur_reward(action_values, action, reward, done,
                                                                        policy, next_state, self.target_policy)
                self.learn_trajectory(action_visits, action_values, state, action, expected_futur_reward)
                memory.forget()