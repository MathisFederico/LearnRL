import numpy as np
from copy import copy
import collections.abc as collections


class Memory():

    """
    A general memory for reinforcement learning agents

    Using the methods :meth:`remember` and :meth:`forget`
    any :Class:`Agent` have a standardized :class:`Memory` !
    
    Attributes
    ----------
        max_memory_len: :class:`int`
            Max number of experiences stocked by the :class:`Memory`
        datas: :class:`dict`
            The dictionary of experiences as :class:`numpy.ndarray`
        MEMORY_KEYS:
            | The keys of core parameters to gather from experience
            | ('observation', 'action', 'reward', 'done', 'next_observation', 'info')
    """

    def __init__(self, max_memory_len=10000): 
        self.MEMORY_KEYS = ('observation', 'action', 'reward', 'done', 'next_observation', 'info')
        self.datas = {key:None for key in self.MEMORY_KEYS}
        self.max_memory_len = max_memory_len

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        """ Add the new experience into the memory forgetting long past experience if neccesary
        
        Parameters
        ----------
            observation:
                The observation given by the |gym.Env| or transformed by an :class:`Agent` hash function
            action:
                The action given by to |gym.Env| or transformed by an :class:`Agent` hash function
            reward: :class:`float`
                The reward given by the |gym.Env|
            done: :class:`bool`
                Whether the |gym.Env| had ended after the action
            next_observation:
                The next_observation given by the |gym.Env| or transformed by the :class:`Agent` hash function
            info: :class:`dict`
                Additional informations given by the |gym.Env|
            **kwargs:
                Optional additional stored informations
        
        """

        def _remember_key(datas, key, value, max_memory_len=self.max_memory_len):
            if datas[key] is None:
                datas[key] = value
            else:
                if len(datas[key]) < max_memory_len:
                    datas[key] = np.concatenate((datas[key], value), axis=0)
                else:
                    datas[key] = np.roll(datas[key], shift=-1, axis=0)
                    datas[key][-1] = np.array(value)

        for key, value in zip(self.MEMORY_KEYS, (observation, action, reward, done, next_observation, info)):
            # Check that value is an instance of numpy.ndarray or transform the value
            if type(value) == np.ndarray:
                value = value[np.newaxis, ...]
            if isinstance(value, collections.Sequence) or type(value) != np.ndarray:
                value = np.array([value])
            _remember_key(self.datas, key, value)
        
        # Add optional supplementary parameters
        for key in param:
            _remember_key(self.datas, key, param[key])


    def sample(self, sample_size=0, method='naive_uniform', return_copy=True):
        """ Return a sample of experiences stored in the memory
        
        Parameters
        ----------
            sample_size: int
                The size of the sample to get from memory, if 0 return all memory.
            method: str
                On of ("last", "naive_uniform", "uniform"). The sampling method.
            copy: bool
                If True, return a copy of the memory sampled.
        
        Return
        ------
            datas: list
                The list of :class:`numpy.ndarray` of memory samples for each key in MEMORY_KEYS.
        
        """
        if method not in ['naive_uniform', 'last']:
            raise NotImplementedError(f'Method {method} is not implemented yet')
        
        n_experiences = len(self.datas['observation'])

        if n_experiences <= sample_size or sample_size == 0:
            datas = [self.datas[key] for key in self.MEMORY_KEYS]
        else:
            if method == 'naive_uniform':
                sample_indexes = np.random.choice(np.arange(n_experiences), size=sample_size)
            elif method == 'last':
                sample_indexes = np.arange(n_experiences - sample_size, n_experiences)
            datas = [self.datas[key][sample_indexes] for key in self.MEMORY_KEYS]

        if return_copy:
            datas = [copy(value) for value in datas]

        return datas

    def forget(self):
        """ Remove all memory"""
        self.datas = {key:None for key in self.MEMORY_KEYS}

