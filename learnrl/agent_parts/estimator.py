# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import numpy as np
from copy import deepcopy
import gym.spaces as spaces

class Estimator():

    """ Estimator base object

    The methods build, fit and predict must be specified.
    Kwargs are passed to the build method.

    Arguments
    ---------
        observation_space: |gym.Space|
            The observation space of the agent
        action_space: |gym.Space|
            The action space of the agent
        learning_rate: float
            The learning rate of the estimator
        learning_rate_decay: float
            The learning rate decay of the estimator
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

    def __init__(self, observation_space, action_space, learning_rate=0.1, learning_rate_decay=0, verbose=0,**kwargs):
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
        self.fit(observations, actions, Y)
    
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
        else:
            raise TypeError(f'Space {type(space)} is not supported yet ... open an issue if needed')

    @staticmethod
    def _get_space_shape(space):
        if isinstance(space, spaces.Discrete):
            return (space.n,)
        elif isinstance(space, spaces.MultiDiscrete):
            return space.nvec.shape
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

class TableEstimator(Estimator):

    def build(self, **kwargs):
        dtype = kwargs.get('dtype', None)
        self.name = 'table'
        self.observation_encoder, self.observation_decoder = self._get_table_encoder_decoder(self.observation_space)
        self.action_encoder, self.action_decoder = self._get_table_encoder_decoder(self.action_space)
        self.table = np.zeros((self.observation_size, self.action_size), dtype=dtype)
    
    def load(self, data):
        self.table = data

    def fit(self, observations, actions, Y):
        delta = Y - self.table[observations, actions]
        self.table[observations, actions] += self.learning_rate * delta
    
    def predict(self, observations, actions):
        if actions is not None:
            return self.table[observations, actions]
        else:
            return self.table[observations, :]
        
    def __str__(self):
        return str(self.table)

class KerasEstimator(Estimator):

    def __init__(self, observation_space, action_space,
                       learning_rate=0.1, learning_rate_decay=1,
                       epochs_per_step=1, batch_size=32,
                       freezed_steps=0, **kwargs):
        self.model = None
        super().__init__(observation_space, action_space, learning_rate, learning_rate_decay, **kwargs)
        self.action_encoder, self.action_decoder = self._get_table_encoder_decoder(self.action_space)
        self.freezed_steps = freezed_steps
        if self.freezed_steps > 0:
            self.step_freezed_left = freezed_steps
            self.model_freezed = deepcopy(self.model)
        self.name = 'keras'
        self.batch_size = batch_size
        self.epochs = epochs_per_step

    def build(self, **kwargs):
        raise NotImplementedError

    def preprocess(self, observations, actions=None):
        raise NotImplementedError

    def fit(self, observations, actions, Y):
        x_train = self.preprocess(observations, actions)
        y_train = self.model.predict(x_train)
        y_train[:, actions] = Y
        self.model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)

        if self.freezed_steps > 0:
            if self.step_freezed_left == 0:
                self.model_freezed.set_weights(self.model.get_weights()) 
                self.step_freezed_left = self.freezed_steps
                if self.verbose > 0:
                    print("Freezed model updated")
            self.step_freezed_left -= 1

    def predict(self, observations, actions=None):
        x = self.preprocess(observations, actions)
        if self.freezed_steps > 0:
            Y = self.model_freezed.predict(x)
        else:
            Y = self.model.predict(x)
        if actions is not None:
            return Y[actions]
        else:
            return Y

