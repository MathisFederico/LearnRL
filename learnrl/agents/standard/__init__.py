# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from learnrl.evaluation import QLearning
from learnrl.control import Greedy
from learnrl.estimators import TableEstimator
from learnrl import Agent
from learnrl.memory import Memory

from gym import spaces
import numpy as np

import time

class StandardAgent(Agent):

    """  A standard structure of RL agents

    Build with parts :class:`~learnrl.agent_parts.control.Control` for policy,
    :class:`~learnrl.agent_parts.evaluation.Evaluation` for futur rewards estimations
    and :class:`~learnrl.agent_parts.estimator.Estimator` for action_value estimation.
    
    Parameters
    ----------
        observation_space: |gym.Space| 
            The observation_space of the environement that the agent will observe
        action_space: |gym.Space|
            The action_space of the environement that the agent will act on
        control: :class:`~learnrl.control.Control`
            Control object to define policy from :attr:`action_value` *(default is 0.1-Greedy)*
        evaluation:  :class:`~learnrl.evaluation.Evaluation`, optional
            Evaluation object to update :attr:`action_value` from agent :class:`~learnrl.core.Memory` *(default is QLearning)*
        action_values: :class:`~learnrl.estimators.Estimator`, optional
            Known as Q(s,a), this represent the expected return (futur rewards) given that
            the agent took the action a in the state s. *(default is a Table)*
        action_visits: :class:`~learnrl.estimators.Estimator`, optional
            Known as N(s,a), this represent the number of times that
            the agent took the action a in the state s.
        online: bool, optional
            If False, wait the end of the episode to learn, else learn every step.
        sample_size: int, optional
            Maximum number of samples to take from memory to perform a learning step. If 0, takes all memory.
        sample_method: str, optional
            The sampling used, see :meth:`~learnrl.core.Memory.sample` for details.
        forget_after_update: bool, optional
            If True, forget all past memory after a learning step.
        step_skip: int, optional
            The number of steps between learn calls.
    
    Attributes
    ----------
        All args and kwargs becomes attributes.

        name: str
            The name of the DeepRLAgent

    """
    
    def __init__(self, observation_space, action_space, control=None, evaluation=None, action_values=None, action_visits=None, **kwargs):
        memory_len = kwargs.pop('memory_len', 10000)
        self.memory = Memory(memory_len)

        self.online = kwargs.pop('online', True)
        self.control = control if control is not None else Greedy(**kwargs)
        self.evaluation = evaluation if evaluation is not None else QLearning(online=self.online, **kwargs)

        self.name = f'standard_{self.control.name}_{self.evaluation.name}_{kwargs}'

        self.action_values = action_values if action_values is not None else TableEstimator(observation_space, action_space, **kwargs)
        self.action_visits = action_visits
        if action_visits is None:
            if self.control.need_action_visit:
                self.action_visits = TableEstimator(observation_space, action_space, learning_rate=1, dtype=np.uint64)

        self.sample_size = kwargs.pop('sample_size', 128)
        self.forget_after_update = kwargs.pop('forget_after_update', isinstance(self.action_values, TableEstimator))
        default_sample_method = 'naive_uniform' if self.online else 'last'
        self.sample_method = kwargs.pop('sample_method', default_sample_method)

        self.observation_space = observation_space
        self.action_space = action_space

        self.step_skip = kwargs.pop('step_skip', 0)
        self._step_to_skip = self.step_skip
    
    def act(self, observation, greedy=False):
        """ Gives the agent action when an observation is given.
        
        This function defines the agent behavior.

        Parameters
        ----------
            observation: :class:`numpy.ndarray` or int
                The observation given by the |gym.Env| step to the agent
            greedy: bool
                Whether the agent will act greedly or not (turn off exploration)
            
        Return
        ------
            action_taken: sample of |gym.Space|
                The action taken by the agent
        """
        observation = self.action_values.observation_encoder(observation)
        if isinstance(observation, np.ndarray):
            observation = observation[np.newaxis, :]
        else: observation = np.array([observation])
        
        policy = self.control._get_policy(observation, self.action_values, self.action_visits, greedy=greedy)[0]
        action_id = np.random.choice(range(len(policy)), p=policy)
        action_taken = self.action_values.action_decoder(action_id)
        
        if action_taken not in self.action_space:
            raise ValueError(f'Action taken should be in action_space, but {action_taken} was not in {self.action_space}. '
                             f'Check the action_decoder of your action_values Estimator')
        return action_taken
    
    def remember(self, observation, action, reward, done, next_observation=None, info={}):
        self.memory.remember(self.action_values.observation_encoder(observation),
                             self.action_values.action_encoder(action),
                             reward,
                             done,
                             self.action_values.observation_encoder(next_observation),
                             info)

    def learn(self, target_control=None):
        logs = {}

        # Skip learning step if step_skip > 0
        skip = self._skip()
        logs.update({'skip': skip})
        if skip:
            return

        # Take a sample of experiences
        observations, actions, reward, done, next_observation, _ = self.memory.sample(sample_size=self.sample_size, method=self.sample_method)

        # Do the online/offline learning
        if self.online or done[-1]:
            # Get expected rewards from self.evaluation
            target_control = target_control if target_control is not None else self.control
            logs.update({'target_control': target_control})

            expected_rewards = self.evaluation._get_evaluation(reward, done, next_observation, self.action_values, self.action_visits, target_control=target_control)

            # Update estimators
            action_values_logs = self.action_values._fit(observations, actions, Y=expected_rewards)
            logs.update({'action_value': action_values_logs})
            if self.action_visits: 
                action_visits_logs = self.action_visits._fit(observations, actions, Y=self.action_visits(observations, actions)+1)
                logs.update({'action_visit': action_visits_logs})
        
            # Forget memory if needed
            logs.update({'forget': self.forget_after_update})
            if self.forget_after_update:
                self.memory.forget()
        
            # Update hyperparameters
            self.control.update_exploration()
            logs.update({'exploration': self.control.exploration})
            self.action_values.update_learning_rate()
            logs.update({'learning_rate': self.action_values.learning_rate})
        
        return logs
        
    def _skip(self):
        if self.step_skip > 0:
            if self._step_to_skip > 0:
                self._step_to_skip -= 1
                return True
            else:
                self._step_to_skip = self.step_skip
                return False
    
    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name
    
    def __call__(self, observation, greedy=False):
        return self.act(observation, greedy=greedy)
