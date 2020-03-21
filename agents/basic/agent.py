from agents.basic.evaluation import MonteCarlo, TemporalDifference
from agents.basic.control import Greedy 
from agents.agent import Agent

from gym import spaces
from warnings import warn
import numpy as np

class BasicAgent(Agent):

    """ 
    A general structure for table-based RL agents.
    You can use different evaluation and control methods.
    
    Evaluations : agents.basic.evaluation
        'mc', 'montecarlo' -> Monte-Carlo evaluation
        'td', 'tempdiff' -> TemporalDifference evaluation
        
    Control : agents.basic.control
        'greedy' -> epsilon_greedy with epsilon=exploration
        'ucb' -> ucb with c=exploration
        'puct' -> puct with c=exploration
    """

    name = 'basic'
    
    def __init__(self, state_space, action_space, control=None, evaluation=None, **kwargs):
        
        super().__init__()

        self.state_size, self._hash_state, self._invert_hash_state = self.get_size_and_hash(state_space)
        self.action_size, self._hash_action, self._invert_hash_action = self.get_size_and_hash(action_space)

        self.control = control if control is not None else Greedy(self.action_size, **kwargs)
        self.evaluation = evaluation if evaluation is not None else MonteCarlo(**kwargs)

        self.name = f'{self.name}_{self.control.name}_{self.evaluation.name}_{kwargs}'
    
        self.action_values = np.zeros((self.state_size, self.action_size))
        self.action_visits = np.zeros((self.state_size, self.action_size))

        self.action_space = action_space
    
    def act(self, state, greedy=False):
        state_id = self._hash_state(state)

        policy = self.control.get_policy(state_id, self.action_values, self.action_visits)
        action_id = np.random.choice(range(self.action_size), p=policy)

        action_taken = self._invert_hash_action(action_id)
        assert action_taken in self.action_space

        return action_taken
    
    def remember(self, state, action, reward, done, next_state=None, info={}):
        self.memory.remember(self._hash_state(state), self._hash_action(action), reward, done, self._hash_state(next_state), info)

    def learn(self, **kwargs):
        self.evaluation.learn(action_values=self.action_values, action_visits=self.action_visits,
                              memory=self.memory, control=self.control, **kwargs)
        self.control.update_exploration()
        self.evaluation.update_learning_rate()

    def get_size_and_hash(self, space):

        int_id = lambda x: int(x)

        if isinstance(space, spaces.Discrete):
            return space.n, int_id, int_id
        elif isinstance(space, spaces.MultiDiscrete):
            base_mat = np.ones_like(space.nvec, dtype=np.uint32)
            rank = base_mat.ndim
            p = 1
            if rank == 0:
                return space.nvec, int_id, int_id
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
                return np.sum(space_sample*base_mat)
            
            def invert_hash_multidiscrete(hashed_space_sample, base_mat=base_mat):
                flat_mat = base_mat.flatten()
                space_sample = np.zeros_like(flat_mat)
                for i in range(len(space_sample))[::-1]:
                    space_sample[i] = hashed_space_sample // flat_mat[i]
                    hashed_space_sample -= space_sample[i] * flat_mat[i]
                return space_sample.reshape(base_mat.shape)

            return np.prod(space.nvec), hash_multidiscrete, invert_hash_multidiscrete
        else:
            raise TypeError(f'Agent {self.name} cannot handle the space of type {type(space)}')

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
    
    def __call__(self, state, greedy=False):
        return self.act(state, greedy=False)


class QLearningAgent(BasicAgent):

    def __init__(self, state_space, action_space, control=None, evaluation=None, **kwargs):

        if evaluation:
            raise ValueError(f"'evaluation' argument shouldn't be specified for QLearningAgent (Forced TemporalDifference) but was set to {evaluation}")
        target_control = kwargs.get('target_control')
        if target_control:
            raise ValueError(f"'target_control' keyword argument shouldn't be specified for QLearningAgent (Forced Greedy) but was set to {target_control}")
        online = kwargs.get('online')
        if online is not None:
            if online != True:
                raise ValueError(f"'online' argument shouldn't be specified for QLearningAgent (Forced online=True) but was set to {online}")
        
        super().__init__(state_space, action_space, control=control, **kwargs)

        kwargs['online'] = True
        self.evaluation = TemporalDifference(target_control=Greedy(self.action_size, initial_exploration=0), **kwargs)
        self.name = f'qlearning_{self.control.name}_{kwargs}'

