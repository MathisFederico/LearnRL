from agents.basic.evaluation import MonteCarlo, TemporalDifference, QLearning
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

        self.state_size, self._hash_state = self.get_size_and_hash(state_space)
        self.action_size, self._hash_action = self.get_size_and_hash(action_space)

        self.control = control if control is not None else Greedy(self.action_size, **kwargs)
        self.evaluation = evaluation if evaluation is not None else QLearning(**kwargs)

        self.name = f'{self.name}_{self.control.name}_{self.evaluation.name}_{kwargs}'
    
        self.action_values = np.zeros((self.state_size, self.action_size))
        self.action_visits = np.zeros((self.state_size, self.action_size))

        self.action_space = action_space
    
    def act(self, state, legal_actions, greedy=False):
        state_id = self._hash_state(state)
        policy = self.control.get_policy(state_id, self.action_values, self.action_visits)

        if len(legal_actions) == 0:
            action_taken = self.action_space.sample()
            warn('No legal action left !\nTook random sample from action space hoping for the environement to handle it !')
            return action_taken

        legal_actions_id = np.array([self._hash_action(action) for action in legal_actions])
        legal_policy = policy[legal_actions_id]
        legal_policy += (1 - np.sum(legal_policy))/len(legal_policy) # TO CHANGE TO SOFTMAX
        action_id = np.random.choice(range(len(legal_actions)), p=legal_policy)
        action_taken = legal_actions[action_id]
        return action_taken
    
    def remember(self, state, action, reward, done, next_state=None, info={}):
        self.memory.remember(self._hash_state(state), self._hash_action(action), reward, done, self._hash_state(next_state), info)

    def learn(self, **kwargs):
        self.evaluation.learn(action_values=self.action_values, action_visits=self.action_visits,
                              memory=self.memory, control=self.control, **kwargs)
        self.control.update_exploration()
        self.evaluation.update_learning_rate()

    def get_size_and_hash(self, space):
        if isinstance(space, spaces.Discrete):
            return space.n, lambda space_sample: int(space_sample)
        elif isinstance(space, spaces.MultiDiscrete):
            base_mat = np.ones_like(space.nvec, dtype=np.uint32)
            rank = base_mat.ndim
            p = 1
            if rank == 0:
                return space.nvec, lambda space_sample: int(space_sample)
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
                raise ValueError(f'Arrays of rank {rank} are not supported')
            
            def hash_multidiscrete(space_sample, base_mat=base_mat):
                return np.sum(space_sample*base_mat)

            return np.prod(space.nvec), hash_multidiscrete
        else:
            raise TypeError(f'Agent {self.name} cannot handle the space of type {type(space)}')

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
    
    def __call__(self, state, legal_actions, greedy=False):
        return self.act(state, legal_actions, greedy=False)
