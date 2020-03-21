"""
Evaluation methodes to modify the value fonctions from experiences
"""
from agents.agent import Memory
from agents.basic.control import Control
import numpy as np

class Evaluation():

    """
    Basic evaluation object\n
    This method must be specified : learn(self, action_values, memory, learning_rate, action_visits=None).
    """

    def __init__(self, initial_learning_rate=0.1, name=None, **kwargs):
        self.learning_rate = initial_learning_rate
        
        if name is None:
            raise ValueError("The Control Object must have a name")
        self.name = name
        self.decay = kwargs.get('learning_rate_decay', 1)

    def learn(self, action_visits, action_values, memory:Memory, **kwargs):
        raise NotImplementedError
    
    def update_learning_rate(self, learning_rate=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate *= self.decay

    def __str__(self):
        return self.name


class MonteCarlo(Evaluation):

    def __init__(self, initial_learning_rate=0.1, **kwargs):
        super().__init__(initial_learning_rate=initial_learning_rate, name="mc", **kwargs)

    def learn(self, action_visits, action_values, memory:Memory, **kwargs):
        datas = memory.datas

        if np.any(datas['done']):
            total_return = np.sum(datas['reward'])

            action_visits[datas['state'], datas['action']] += 1
            delta = total_return - action_values[datas['state'], datas['action']]
            action_values[datas['state'], datas['action']] += self.learning_rate * delta

            memory.forget()


class TemporalDifference(Evaluation):

    def __init__(self, initial_learning_rate=0.1, **kwargs):
        super().__init__(initial_learning_rate=initial_learning_rate, name="td", **kwargs)
        self.target_control = kwargs.get('target_control')
        self.target_policy = self.target_control.get_policy if self.target_control else None
        self.online = kwargs.get('online', False)

    @staticmethod
    def get_next_legal_actions(next_state, memory):
        if next_state in memory.legal_actions:
            return memory.legal_actions[next_state]
        else:
            raise KeyError(f'Couldn\'t find state {next_state} legal_action\'s in agent memory')

    @staticmethod
    def get_expected_futur_reward(action_visits, action_values, action, reward, done, policy, next_state=None, target_policy=None):
        expected_futur_reward = reward.astype(np.float64)
        not_done = np.logical_not(done)
        if target_policy is None:
            action_impact = policy(next_state[not_done], action_values, action_visits) 
        else:
            action_impact = target_policy(next_state[not_done], action_values, action_visits)
        
        expected_futur_reward[not_done] += np.sum(action_impact * action_values[next_state[not_done], :], axis=-1)
        return expected_futur_reward
    
    def learn_trajectory(self, action_visits, action_values, state, action, expected_futur_reward):
        action_visits[state, action] += 1
        delta = expected_futur_reward - action_values[state, action]
        action_values[state, action] += self.learning_rate * delta

    def learn(self, action_visits, action_values, memory:Memory, control:Control, **kwargs):

        # Get specific parameters for TD
        policy = control.get_policy
        if policy is None:
            raise ValueError('You must specify a policy for TD evaluation')
        datas = memory.datas

        # If Online learning, learns every step
        state, action, reward, done, next_state, _ = [datas[key] for key in memory.MEMORY_KEYS]
        if self.online:
            # If done, we update value fonctions toward real reward
            if np.any(done):
                self.learn_trajectory(action_visits, action_values, state, action, reward)
            # If not done, we estimate futur rewards based on memory and value fonctions
            else:
                expected_futur_reward = self.get_expected_futur_reward(action_visits, action_values, action, reward, done, policy,
                                                                        next_state, self.target_policy)
                self.learn_trajectory(action_visits, action_values, state, action, expected_futur_reward)
            memory.forget()
        
        # If Offline learning, learns in a batch at the end of the episode
        else:
            if np.any(done):
                expected_futur_reward = self.get_expected_futur_reward(action_visits, action_values, action, reward, done,
                                                                        policy, next_state, self.target_policy)
                self.learn_trajectory(action_visits, action_values, state, action, expected_futur_reward)
                memory.forget()