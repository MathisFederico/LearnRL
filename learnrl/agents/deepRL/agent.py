# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>
""" DeepRL Agents

"""

from learnrl.agents.deepRL.evaluation import QLearning
from learnrl.agents.deepRL.control import Greedy
from learnrl.agents.deepRL.estimator import TableEstimator
from learnrl.core import Agent

from gym import spaces
import numpy as np

class DeepRLAgent(Agent):

    """  A general structure for deepRL agents 
    using an :class:`~learnrl.agents.deepRL.evaluation.Evaluation` 
    and a :class:`~learnrl.agents.deepRL.control.Control`.
    
    Parameters
    ----------
        observation_space: |gym.Space| 
            The observation_space of the environement that the agent will observe
        action_space: |gym.Space|
            The action_space of the environement that the agent will act on
        control: :class:`~learnrl.agents.deepRL.control.Control`
            Control object to define policy from :attr:`action_value` (default is 0.1-Greedy)
        evaluation: :class:`~learnrl.agents.deepRL.evaluation.Evaluation`
            Evaluation object to update :attr:`action_value` from agent :class:`~learnrl.core.Memory` (default is QLearning)
    
    Attributes
    ----------
        name: str
            The name of the DeepRLAgent
        action_values: :class:`Estimator`
            Known as Q(s,a), this represent the expected return (futur rewards) given that
            the agent took the action a in the state s.
        action_visits: :class:`Estimator`
            Known as N(s,a), this represent the number of times that
            the agent took the action a in the state s.
    """

    name = 'deeprl'
    
    def __init__(self, observation_space, action_space, control=None, evaluation=None, action_values=None, action_visits=None, **kwargs):
        
        super().__init__()
        self.control = control if control is not None else Greedy(**kwargs)
        self.evaluation = evaluation if evaluation is not None else QLearning(**kwargs)

        self.name = f'{self.name}_{self.control.name}_{self.evaluation.name}_{kwargs}'
    
        self.action_values = action_values if action_values is not None else TableEstimator(observation_space, action_space)
        self.action_visits = action_visits
        if action_visits is None:
            if not isinstance(self.control, Greedy):
                self.action_visits = TableEstimator(observation_space, action_space, learning_rate=1, dtype=np.uint64)

        self.observation_space = observation_space
        self.action_space = action_space
    
    def act(self, observation, greedy=False):
        """ Gives the agent action when an observation is given.
        
        This function defines the agent behavior or policy.

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
        if isinstance(observation, np.ndarray):
            observation = observation[np.newaxis, :]
        else: observation = np.array([observation])

        policy = self.control.get_policy(observation, self.action_values, self.action_visits)[0]
        action_id = np.random.choice(range(len(policy)), p=policy)
        action_taken = self.action_values.action_decoder(action_id)
        
        if action_taken not in self.action_space:
            raise ValueError(f'Action taken should be in action_space, but {action_taken} was not in {self.action_space}. '
                             f'Check the action_decoder of your action_values Estimator')
        return action_taken
    
    def remember(self, observation, action, reward, done, next_observation=None, info={}):
        self.memory.remember(observation, action, reward, done, next_observation, info)

    def learn(self, **kwargs):
        observations, actions, expected_rewards = self.evaluation.eval(self.action_values, self.action_visits, self.memory, self.control, **kwargs)
        self.action_values.fit(observations=observations, actions=actions, Y=expected_rewards)
        if self.action_visits: self.action_visits.fit(observations=observations, actions=actions, Y=self.action_visits(observations, actions)+1)
        self.control.update_exploration()
        self.action_values.update_learning_rate()
    
    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
    
    def __call__(self, observation, greedy=False):
        return self.act(observation, greedy=False)
