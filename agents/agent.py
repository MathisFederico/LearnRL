import numpy as np
import collections.abc as collections

class Memory():

    MEMORY_KEYS = ('state', 'action', 'reward', 'done', 'next_state', 'info')
    max_memory_len = 1e4

    def __init__(self):
        self.datas = {key:None for key in self.MEMORY_KEYS}

    def remember(self, state, action, reward, done, next_state=None, info={}, **param):
        for key, value in zip(self.MEMORY_KEYS, (state, action, reward, done, next_state, info)):

            # Check that value is an instance of numpy.ndarray or transform the value
            if isinstance(value, collections.Sequence) or type(value) != np.ndarray or value.ndim < 1:
                value = np.array([value])

            # Add the new experience into the memory forgetting long past experience if neccesary
            if self.datas[key] is None:
                self.datas[key] = value
            else:
                if len(self.datas[key]) < self.max_memory_len:
                    self.datas[key] = np.concatenate((self.datas[key], value), axis=0)
                else:
                    self.datas[key] = np.roll(self.datas[key], shift=-1, axis=0)
                    self.datas[key][-1] = np.array(value)
        
        # Same for suplementary parameters
        for key in param:
            if self.datas[key] is None:
                self.datas[key] = param[key]
            else:
                if len(self.datas[key]) < self.max_memory_len:
                    self.datas[key] = np.concatenate((self.datas[key], param[key]), axis=0)
                else:
                    self.datas[key] = np.roll(self.datas[key], shift=-1, axis=0)
                    self.datas[key][-1] = np.array(param[key])

    
    def forget(self):
        self.datas = {key:None for key in self.MEMORY_KEYS}


class Agent():

    name = 'DefaultAgent'

    state_values = None
    state_visits = None

    action_values = None
    action_visits = None
    
    memory = Memory()

    def policy(self, observation, legal_actions):
        raise NotImplementedError

    def act(self, observation, legal_actions):
        raise NotImplementedError
    
    @staticmethod
    def _hash_state(state):
        return state

    @staticmethod
    def _hash_action(action):
        return action

    def remember(self, state, action, reward, done, next_state=None, info={}, **param):
        self.memory.remember(self._hash_state(state), self._hash_action(action), reward, done, self._hash_state(next_state), info, **param)
    
    def forget(self):
        self.memory.forget()
    
    def learn(self):
        raise NotImplementedError
    
    def render(self):
        raise NotImplementedError
