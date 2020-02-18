import numpy as np
import collections

class Memory():

    MEMORY_KEYS = ('state', 'action', 'reward', 'done', 'next_state', 'info')
    datas = {key:None for key in MEMORY_KEYS}
    max_memory_len = 1e5

    def remember(self, state, action, reward, done, next_state=None, info={}):
        for key, value in zip(self.MEMORY_KEYS, (state, action, reward, next_state, done, info)):

            # Check that value is an instance of numpy.ndarray or transform the value
            if isinstance(value, collections.Sequence):
                value = np.array(value)
            elif type(value) != np.ndarray:
                value = np.array([value])

            # Add the new experience into the memory forgetting long past experience if neccesary
            if self.datas[key] is None:
                self.datas[key] = value
            else:
                if len(self.datas[key]) <= self.max_memory_len:
                    self.datas[key] = np.concatenate((self.datas[key], value))
                else:
                    self.datas[key] = np.roll(self.datas[key], shift=-1, axis=0)
                    self.datas[key][-1] = np.array(value)
    
    def forget(self):
        self.datas = {key:None for key in self.MEMORY_KEYS}


class Agent():

    name = 'DefaultAgent'

    state_values = None
    state_visits = None

    action_values = None
    action_visits = None
    
    memory = Memory()

    def policy(self, observation):
        raise NotImplementedError

    def play(self, observation):
        raise NotImplementedError
    
    def remember(self, state, action, reward, done, next_state=None, info={}):
        self.memory.remember(state, action, reward, done, next_state, info)
    
    def forget(self):
        self.memory.forget()
    
    def learn(self):
        raise NotImplementedError
    
    def render(self):
        raise NotImplementedError
