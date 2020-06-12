# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import numpy as np
import collections.abc as collections
from copy import copy
import warnings

from time import time
from gym import Env

from learnrl.callbacks import CallbackList, Logger


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
        
        # Add optional suplementary parameters
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


class Agent():

    """ A general structure for reinforcement learning agents    
    
    It uses by default a :class:`Memory`

    Attributes
    ----------

        name: :class:`str`
            The Agent's name
        memory: :class:`Memory`
            The Agent's memory
    
    """

    def act(self, observation, greedy=False):
        """ How the :ref:`Agent` act given an observation
        
        Parameters
        ----------
            observation:
                The observation given by the |gym.Env|
            greedy: bool
                If True, act greedely (without exploration)

        """
        raise NotImplementedError

    def learn(self):
        """ How the :ref:`Agent` learns from his experiences """
        raise NotImplementedError

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        """ Uses the agent's :class:`Memory` to remember experiences
        
        Often, the agent will use a |hash| to store observations efficiently

        Example
        -------
            >>>  self.memory.remember(self.observation_encoder(observation),
            ...                       self.action_encoder(action),
            ...                       reward, done, 
            ...                       self.observation_encoder(next_observation), 
            ...                       info, **param)
        """
        raise NotImplementedError
    

class MultiEnv(Env):

    r"""
    A layer over the gym |gym.Env| class able to handle environements with multiple agents.
   
    .. note::
        A :ref:`MultiEnv` must be in a :ref:`Playground` in order to work !

    | The main add in MultiEnv is the method: 
    |   turn

    On top of the main API basic methodes (see |gym.Env|):
        * step: take a step of the |gym.Env| given the action of the active player
        * reset: reset the |gym.Env| and returns the first observation
        * render
        * close 
        * seed

    Attributes
    ----------
        action_space: |gym.Space|
            The Space object corresponding to actions  
        observation_space: |gym.Space|
            The Space object corresponding to observations  
        reward_range: :class:`tuple`
            | A tuple corresponding to the min and max possible rewards.
            | A default reward range set to [-inf,+inf] already exists. 
            | Set it if you want a narrower range.

    """

    def step(self, action):
        """Perform a step of the environement
        
        Parameters
        ----------
            action:
                The action taken by the agent who's turn was given by :meth:`turn`.
        
        Return
        ------
            observation: 
                The observation to give to the :class:`Agent`.
            reward: :class:`float`
                The reward given to the :class:`Agent` for this step.
            done: :class:`bool`
                Is the environement done after this step ?
            info: :class:`dict`
                Additional informations given by the |gym.Env|.
            
        """
        raise NotImplementedError

    def turn(self, state):
        """Give the turn to the next agent to play
    
        Assuming that agents are represented by a list like range(n_player)
        where n_player is the number of players in the game.
        
        Parameters
        ----------
            state: 
                | The real state of the environement.
                | Should be enough to determine which is the next agent to play.

        Return
        ------
            agent_id: :class:`int`
                The next player id

        """
        raise NotImplementedError

    def reset(self):
        """Reset the environement and returns the initial state

        Return
        ------
            observation:
                The observation for the first :class:`Agent` to play
        
        """
        raise NotImplementedError


class Playground():

    """ A playground is used to run agent(s) on an environement.

    Attributes
    ----------
        env: :class:`MultiEnv` or |gym.Env|
            Environement in which agents will play
        agents: list or :class:`Agent`
            List of :class:`Agent` to run on the :class:`MultiEnv`

    """

    def __init__(self, environement:Env, agents):
        assert isinstance(environement, Env)
        if isinstance(agents, Agent):
            agents = [agents]
        for agent in agents:
            assert isinstance(agent, Agent)
        
        self.env = environement
        self.agents = agents

    def run(self, episodes, render=True, learn=True, cycle_len=None, cycle_prop=0.05, verbose=0, callbacks=[]):
        """ Let the agent(s) play on the environement for a number of episodes.
        
        Arguments
        ---------
            episodes: int
                Number of episodes to run.
            render: bool
                If True, call :meth:`MultiEnv.render` every step.
            learn: bool
                If True, call :meth:`Agent.learn` every step.
            cycle_len: int
                Number of episodes that compose a cycle, used in :class:`~learnrl.callbacks.Callback`.
            cycle_prop: float
                Propotion of total episodes to compose a cycle, used if cycle_len is not set.
            verbose: int
                The verbosity level: 0 (silent), 1 (cycle), 2 (episode), 3 (step), 4 (detailed step).
            callbacks: list
                List of :class:`~learnrl.callbacks.Callback` to use in runs.
        
        """
        cycle_len = cycle_len or max(1, int(cycle_prop*episodes))

        params = {
            'episodes': episodes,
            'cycle_len': cycle_len,
            'verbose': verbose,
            'render': render,
            'learn': learn
        }

        callbacks = CallbackList([Logger()] + callbacks)
        callbacks.set_params(params)
        callbacks.set_playground(self)

        logs = {}
        logs.update(params)
        callbacks.on_run_begin(logs)

        for episode in range(episodes):

            if episode % cycle_len == 0:
                callbacks.on_cycle_begin(episode, logs)

            observation = self.env.reset()
            previous = [{'observation':None, 'action':None, 'reward':None, 'done':None, 'info':None} for _ in range(len(self.agents))]
            done = False
            step = 0

            logs.update({'episode':episode})
            callbacks.on_episode_begin(episode, logs)

            while not done:

                if render: self.env.render()

                agent_id = self.env.turn(observation) if isinstance(self.env, MultiEnv) else 0
                     
                prev = previous[agent_id]
                if learn and prev['observation'] is not None:
                    agent.remember(prev['observation'], prev['action'], prev['reward'], prev['done'], observation, prev['info'])
                    agent.learn()
                
                logs.update({'step':step, 'agent_id':agent_id, 'observation':observation})
                callbacks.on_step_begin(step, logs)

                agent = self.agents[agent_id]
                action = agent.act(observation, greedy=not learn)
                next_observation, reward, done , info = self.env.step(action)

                if learn:
                    for key, value in zip(prev, [observation, action, reward, done, info]):
                        prev[key] = value
                    if done:
                        agent.remember(observation, action, reward, done, next_observation, info)
                        agent.learn()
                
                logs.update({'action': action, 'reward':reward, 'done':done, 'next_observation':next_observation})
                logs.update(info)

                callbacks.on_step_end(step, logs)
                step += 1               
                observation = next_observation
            
            callbacks.on_episode_end(episode, logs)
            if (episode+1) % cycle_len == 0 or episode == episodes - 1:
                callbacks.on_cycle_end(episode, logs)
        
        callbacks.on_run_end(logs)

    def fit(self, episodes, **kwargs):
        """Train the agent(s) on the environement for a number of episodes."""
        learn = kwargs.pop('learn', True)
        render = kwargs.pop('render', False)
        if not learn:
            warnings.warn("learn should be True in Playground.fit(), otherwise the agents will not improve", UserWarning)
        if render:
            warnings.warn("rendering degrades heavely computation speed, use CycleRenderCallback to see your agent performence suring training", RuntimeWarning)

        self.run(episodes, render=render, learn=learn, **kwargs)

    def test(self, episodes, **kwargs):
        """Test the agent(s) on the environement for a number of episodes."""
        learn = kwargs.pop('learn', False)
        render = kwargs.pop('render', True)
        verbose = kwargs.pop('verbose', 0)
        if learn:
            warnings.warn("learn should be False in Playground.test(), otherwise the agents will not act greedy and can have random behavior", UserWarning)
        if not render and verbose == 0:
            warnings.warn("you should set verbose > 0 or render=True to see something ...", UserWarning)

        self.run(episodes, render=render, learn=learn, verbose=verbose, **kwargs)

