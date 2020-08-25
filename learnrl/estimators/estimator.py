# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import numpy as np
from copy import deepcopy
import gym.spaces as spaces

class Estimator():

    """ Estimator base object

    The methods must be specified:
     - `build(self, **kwargs) -> None`
     - `fit(self, observations, actions, Y) -> None`
     - `predict(self, observations, actions) -> Y`
      
    Kwargs are passed to the build method.

    Arguments
    ---------
        observation_space: |gym.Space|
            The observation space of the agent.
        action_space: |gym.Space|
            The action space of the agent.
        learning_rate: float
            The learning rate of the estimator.
        learning_rate_decay: float
            The learning rate decay of the estimator.
        verbose: int
            The amount of informations to be logged by the agent, 0 is silent.

    Attributes
    ----------
        name: str
            The name of the Estimator
        observation_space: |gym.Space|
            The observation space of the agent
        observation_size: int
            The size of the observation space
        observation_shape: tuple
            The shape of the observation space
        observation_encoder: func
            The encoder of the observation space (identity by default)
        observation_decoder: func
            The decoder of the observation space (identity by default)
        action_space: |gym.Space|
            The action space of the agent
        action_size: int
            The size of the action space
        action_shape: tuple
            The shape of the action space
        action_encoder: func
            The encoder of the action space (identity by default)
        action_decoder: func
            The decoder of the action space (identity by default)
    

    """

    def __init__(self, observation_space, action_space, learning_rate=0.1, learning_rate_decay=0, step_skip=0, verbose=0,**kwargs):
        self.observation_space = observation_space
        self.action_space = action_space

        self.observation_size = self._get_space_size(observation_space)
        self.observation_shape = self._get_space_shape(observation_space)

        self.action_size = self._get_space_size(action_space)
        self.action_shape = self._get_space_shape(action_space)

        self.observation_encoder, self.observation_decoder = lambda x: x, lambda x: x
        self.action_encoder, self.action_decoder = lambda x: x, lambda x: x

        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.verbose = verbose

        self.name = None
        self.build(**kwargs)
    
    def build(self, **kwargs):
        """ Build the estimator """
        raise NotImplementedError

    def _fit(self, observations, actions, Y):
        if observations is None or actions is None or Y is None :
            return
        return self.fit(observations, actions, Y)
    
    def fit(self, observations, actions, Y):
        """ Fit the estimator
        
        Arguments
        ---------
            observations: np.ndarray
                The sample of observations.
            actions: np.ndarray
                The sample of actions.
            Y:
                The ground truth that the estimator should predict.
        
        """
        raise NotImplementedError
    
    def predict(self, observations, actions):
        """ Uses the estimator to make a prediction

        Arguments
        ---------
            observations: np.ndarray
                The sample of observations.
            actions: np.ndarray
                The sample of actions.       
        
        """
        raise NotImplementedError
    
    def encoder(self, arr, arr_type):
        flatten_array = np.reshape(arr, (arr.shape[0], -1))
        if arr_type == 'action':
            if self.action_encoder:
                return np.apply_along_axis(self.action_encoder, axis=1, arr=flatten_array)
            else:
                return arr
        elif arr_type == 'observation':
            if self.observation_encoder:
                return np.apply_along_axis(self.observation_encoder, axis=1, arr=flatten_array)
            else:
                return arr
        else:
            raise ValueError(f"{arr_type} is an unkwoned type of array, use 'action' or 'observation' instead")

    def update_learning_rate(self, learning_rate=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate *= 1 - self.learning_rate_decay

    @staticmethod
    def _get_space_size(space):
        if isinstance(space, spaces.Discrete):
            return space.n
        elif isinstance(space, spaces.MultiDiscrete):
            return np.prod(space.nvec)
        elif isinstance(space, spaces.Box):
            return np.prod(space.shape)
        else:
            raise TypeError(f'Space {type(space)} is not supported yet ... open an issue if needed')

    @staticmethod
    def _get_space_shape(space):
        if isinstance(space, spaces.Discrete):
            return (space.n,)
        elif isinstance(space, spaces.MultiDiscrete):
            return space.nvec.shape
        elif isinstance(space, spaces.Box):
            return space.shape
        else:
            raise TypeError(f'Space {type(space)} is not supported yet ... open an issue if needed')

    @staticmethod
    def _get_table_encoder_decoder(space):
        int_id = lambda x: int(x)
        
        if isinstance(space, spaces.Discrete):
            return int_id, int_id
        
        elif isinstance(space, spaces.MultiDiscrete):
            base_mat = np.ones_like(space.nvec, dtype=np.uint32)
            rank = base_mat.ndim
            p = 1
            if rank == 0:
                return int_id, int_id
            elif rank == 1:
                for i in range(len(base_mat)):
                    base_mat[i] = p
                    p *= space.nvec[i]
            elif rank == 2:
                for i in range(base_mat.shape[0]): # pylint:disable=E1136
                    for j in range(base_mat.shape[1]): # pylint:disable=E1136
                        base_mat[i, j] = p
                        p *= int(space.nvec[i, j])
            else:
                raise ValueError(f'Arrays of rank {rank} are not supported yet ... open an issue if needed')
            
            def hash_multidiscrete(space_sample, base_mat=base_mat):
                return np.sum(space_sample.flatten()*base_mat.flatten())
            
            def invert_hash_multidiscrete(hashed_space_sample, base_mat=base_mat):
                flat_mat = base_mat.flatten()
                space_sample = np.zeros_like(flat_mat)
                for i in range(len(space_sample))[::-1]:
                    space_sample[i] = hashed_space_sample // flat_mat[i]
                    hashed_space_sample -= space_sample[i] * flat_mat[i]
                return space_sample.reshape(base_mat.shape)

            return hash_multidiscrete, invert_hash_multidiscrete
        else:
            raise TypeError(f'Cannot handle the space of type {type(space)}')

    def __call__(self, observations, actions=None):
        return self.predict(observations, actions)

