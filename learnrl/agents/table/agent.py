# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>
"""
| Table agents are the simplest form of RL Agents.
| With experience, we build an action_value array (Q in literature).
| Q(s,a) being the expected futur rewards given that the agent did the action a ins the state s.
| For that, they are composed of two main objects : :class:`Control` and :class:`Evaluation`

| A :class:`Control` object uses the action_value to determine the probabilities of choosing every action
| An :class:`Evaluation` object uses the experience of the :ref:`TableAgent`
  present in his :ref:`Memory`.

"""

from learnrl.agents.table.evaluation import QLearning
from learnrl.agents.table.control import Greedy
from learnrl.core import Agent

from gym import spaces
import numpy as np

class TableAgent(Agent):

    """  A general structure for table-based RL agents 
    using an :class:`~learnrl.agents.table.evaluation.Evaluation` 
    and a :class:`~learnrl.agents.table.control.Control`.
    
    Parameters
    ----------
        observation_space: |gym.Space| 
            The observation_space of the environement that the agent will observe
        action_space: |gym.Space|
            The action_space of the environement that the agent will act on
        control: :class:`~learnrl.agents.table.control.Control`
            Control object to define policy from :attr:`action_value` (default is 0.1-Greedy)
        evaluation: :class:`~learnrl.agents.table.evaluation.Evaluation`
            Evaluation object to update :attr:`action_value` from agent :class:`~learnrl.core.Memory` (default is QLearning)
    
    Attributes
    ----------
        name: str
            The name of the TableAgent
        action_values: :class:`numpy.ndarray`
            Known as Q(s,a), this represent the expected return (futur rewards) given that
            the agent took the action a in the state s.
        action_visits: :class:`numpy.ndarray`
            Known as N(s,a), this represent the number of times that
            the agent took the action a in the state s.
    """

    name = 'table'
    
    def __init__(self, observation_space, action_space, control=None, evaluation=None, **kwargs):
        
        super().__init__()

        self.observation_size, self._hash_observation, self._invert_hash_observation = self.get_size_and_hash(observation_space)
        self.action_size, self._hash_action, self._invert_hash_action = self.get_size_and_hash(action_space)

        self.control = control if control is not None else Greedy(self.action_size, **kwargs)
        self.evaluation = evaluation if evaluation is not None else QLearning(**kwargs)

        self.name = f'{self.name}_{self.control.name}_{self.evaluation.name}_{kwargs}'
    
        self.action_values = np.zeros((self.observation_size, self.action_size))
        self.action_visits = np.zeros((self.observation_size, self.action_size))

        self.observation_space = observation_space
        self.action_space = action_space
    
    def act(self, observation, greedy=False):
        """ Gives the agent action when an observation is given.
        
        This function is essential to define the agent behavior.

        Parameters
        ----------
            observation: :class:`numpy.ndarray` or int
                The observation given by the |gym.Env| step to the agent
            greedy: bool
                Whether the agent will act greedly or not
            
        Return
        ------
            action_taken: sample of |gym.Space|
                The action taken by the agent
        """
        observation_id = self._hash_observation(observation)

        policy = self.control.get_policy(observation_id, self.action_values, self.action_visits)
        action_id = np.random.choice(range(self.action_size), p=policy)

        action_taken = self._invert_hash_action(action_id)
        if action_taken not in self.action_space:
            raise ValueError(f'Action taken should be in action_space, but {action_taken} was not in {self.action_space}')
        return action_taken
    
    def remember(self, observation, action, reward, done, next_observation=None, info={}):
        self.memory.remember(self._hash_observation(observation), self._hash_action(action), reward, done, self._hash_observation(next_observation), info)

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
    
    def __call__(self, observation, greedy=False):
        return self.act(observation, greedy=False)
