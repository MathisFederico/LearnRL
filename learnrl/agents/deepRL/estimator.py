
import numpy as np
import gym.spaces as spaces

class Estimator():

    def __init__(self, observation_space, action_space, learning_rate=0.1, learning_rate_decay=1, **kwargs):
        self.observation_space = observation_space
        self.action_space = action_space

        self.observation_size = self.get_space_size(observation_space)
        self.observation_shape = self.get_space_shape(observation_space)

        self.action_size = self.get_space_size(action_space)
        self.action_shape = self.get_space_shape(action_space)

        self.observation_encoder, self.observation_decoder = self.get_encoder_decoder(observation_space)
        self.action_encoder, self.action_decoder = self.get_encoder_decoder(action_space)

        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay

        self.build(**kwargs)
    
    def build(self):
        raise NotImplementedError

    def fit(self, observations, actions, Y):
        raise NotImplementedError
    
    def predict(self, observations, actions):
        raise NotImplementedError
    
    def encoder(self, arr, arr_type):
        flatten_array = np.reshape(arr, (arr.shape[0], -1))
        if arr_type == 'actions' or arr_type== 'action':
            return np.apply_along_axis(self.action_encoder, axis=1, arr=flatten_array)
        elif arr_type == 'observations' or arr_type== 'observation':
            return np.apply_along_axis(self.observation_encoder, axis=1, arr=flatten_array)
        else:
            raise ValueError(f"{arr_type} is an unkwoned type of array ... use 'action' or 'observation' instead")

    def update_learning_rate(self, learning_rate=None):
        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate *= self.learning_rate_decay

    @staticmethod
    def get_space_size(space):
        if isinstance(space, spaces.Discrete):
            return space.n
        elif isinstance(space, spaces.MultiDiscrete):
            return np.prod(space.nvec)
        else:
            raise TypeError(f'Space {type(space)} is not supported yet ... open an issue if needed')

    @staticmethod
    def get_space_shape(space):
        if isinstance(space, spaces.Discrete):
            return (space.n,)
        elif isinstance(space, spaces.MultiDiscrete):
            return space.nvec.shape
        else:
            raise TypeError(f'Space {type(space)} is not supported yet ... open an issue if needed')

    @staticmethod
    def get_encoder_decoder(space):
        int_id = np.vectorize(lambda x: int(x))

        if isinstance(space, spaces.Discrete):
            return int_id, int_id
        
        elif isinstance(space, spaces.MultiDiscrete):
            rank = space.nvec.ndim
            if rank == 0:
                return int_id, int_id
            else:
                flat_vec = space.nvec.flatten()
                base_mat = np.ones_like(flat_vec, dtype=np.uint32)
                p = 1
                for i in range(len(flat_vec)):
                    base_mat[i] = p
                    p *= int(flat_vec[i])
            
            def hash_multidiscrete(space_sample, base_mat=base_mat):
                flatten_sample = space_sample.flatten()
                return np.sum(flatten_sample*base_mat)
            
            def invert_hash_multidiscrete(hashed_space_sample, base_mat=base_mat, sample_shape=space.nvec.shape):
                flat_sample = np.zeros_like(base_mat)
                for i in range(len(flat_sample))[::-1]:
                    flat_sample[i] = hashed_space_sample // base_mat[i]
                    hashed_space_sample -= flat_sample[i] * base_mat[i]
                return flat_sample.reshape(sample_shape)

            return hash_multidiscrete, invert_hash_multidiscrete
        else:
            raise TypeError(f'Space {type(space)} is not supported yet ... open an issue if needed')

    def __call__(self, observations, actions=None):
        return self.predict(observations, actions)

class TableEstimator(Estimator):

    def build(self, **kwargs):
        dtype = kwargs.get('dtype', None)
        self.table = np.zeros((self.observation_size, self.action_size), dtype=dtype)

    def fit(self, observations, actions, Y):
        observations_id = self.observation_encoder(observations)
        actions_id = self.action_encoder(actions)

        delta = Y - self.table[observations_id, actions_id]
        self.table[observations_id, actions_id] += self.learning_rate * delta
    
    def predict(self, observations, actions):
        
        observations_ids = self.encoder(observations, arr_type='observation')
        if actions is not None: 
            action_ids = self.encoder(actions, arr_type='action')
            return self.table[observations_ids, action_ids]
        else:
            return self.table[observations_ids, :]

class KerasEstimator(Estimator):

    def build(self):
        self.model = None
        raise NotImplementedError

    def preprocess(self, observations):
        raise NotImplementedError

    def fit(self, observations, actions, Y, batch_size=1):
        x_train = self.preprocess(observations)
        y_train = self.model.predict(x_train)
        action_ids = self.encoder(actions, arr_type='action')
        y_train[:, action_ids] = Y
        self.model.fit(x_train, y_train, verbose=0)

    def predict(self, observations, actions=None):
        x = self.preprocess(observations)
        Y = self.model.predict(x)

        if actions is not None:
            action_id = self.encoder(actions, arr_type='action')
            return Y[action_id]
        else:
            return Y

