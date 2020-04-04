
import numpy as np
import gym.spaces as spaces

class Estimator():

    def __init__(self, observation_space, action_space, learning_rate=0.1, learning_rate_decay=1, **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space

        self.observation_size = self.get_space_size(observation_space)
        self.action_size = self.get_space_size(action_space)

        self.observation_encoder, self.observation_decoder = self.get_encoder_decoder(observation_space)
        self.action_encoder, self.action_decoder = self.get_encoder_decoder(action_space)

        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        self.build(**kwargs)
    
    def build(self):
        raise NotImplementedError

    def fit(self, observation, action, Y):
        raise NotImplementedError
    
    def predict(self, observation, action):
        raise NotImplementedError

    @staticmethod
    def get_space_size(space):
        if isinstance(space, spaces.Discrete):
            return space.n
        elif isinstance(space, spaces.MultiDiscrete):
            return np.prod(space.nvec)
        else:
            raise TypeError(f'Space {type(space)} is not supported yet ... open an issue if needed')

    @staticmethod
    def get_encoder_decoder(space):
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
                    p *= int(space.nvec[i])
            elif rank == 2:
                for i in range(base_mat.shape[0]): # pylint:disable=E1136
                    for j in range(base_mat.shape[1]): # pylint:disable=E1136
                        base_mat[i, j] = p
                        p *= int(space.nvec[i, j])
            else:
                raise ValueError(f'MultiDiscrete spaces of rank {rank} are not supported yet ... open an issue if needed')
            
            def hash_multidiscrete(space_sample, base_mat=base_mat):
                return np.sum(space_sample*base_mat)
            
            def invert_hash_multidiscrete(hashed_space_sample, base_mat=base_mat):
                flat_mat = base_mat.flatten()
                space_sample = np.zeros_like(flat_mat)
                for i in range(len(space_sample))[::-1]:
                    space_sample[i] = hashed_space_sample // flat_mat[i]
                    hashed_space_sample -= space_sample[i] * flat_mat[i]
                return space_sample.reshape(base_mat.shape)

            return hash_multidiscrete, invert_hash_multidiscrete
        else:
            raise TypeError(f'Space {type(space)} is not supported yet ... open an issue if needed')

    def __call__(self, observation, action=None):
        return self.predict(observation, action)

class TableEstimator(Estimator):

    def build(self, **kwargs):
        dtype = kwargs.get('dtype', None)
        self.table = np.zeros((self.observation_size, self.action_size), dtype=dtype)

    def fit(self, observation, action, Y):
        observation_id = self.observation_encoder(observation)
        action_id = self.action_encoder(action)

        delta = Y - self.table[observation_id, action_id]
        self.table[observation_id, action_id] += self.learning_rate * delta
    
    def predict(self, observation, action):
        observation_id = self.observation_encoder(observation)
        if action is not None: 
            action_id = self.action_encoder(action)
            return self.table[observation_id, action_id]
        else:
            return self.table[observation_id, :]

