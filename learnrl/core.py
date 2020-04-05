# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import numpy as np
import collections.abc as collections

from time import time
from gym import Env


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

    MEMORY_KEYS = ('observation', 'action', 'reward', 'done', 'next_observation', 'info')

    def __init__(self, max_memory_len=10000):
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

        def remember_key(datas, key, value, max_memory_len=self.max_memory_len):
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
            remember_key(self.datas, key, value)
        
        # Add optional suplementary parameters
        for key in param:
            remember_key(self.datas, key, param[key])

    def forget(self):
        """ Remove all memory"""
        self.datas = {key:None for key in self.MEMORY_KEYS}


class Agent():

    """ A general structure for reinforcement learning agents    
    
    It uses by default a :class:`Memory`
    
    """

    name = None    
    memory = Memory()

    def act(self, observation):
        """ How the agent act given an observation
        
        Parameters
        ----------
            observation:
                The observation given by the |gym.Env|

        """
        raise NotImplementedError

    def learn(self):
        """ How the learns from his experiences """
        raise NotImplementedError

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        """ Uses the agent's :class:`Memory` to remember experiences
        
        Often, the agent will use a |hash| to store observations efficiently

        Example
        -------
            >>>  self.memory.remember(self._hash_observation(observation),
            ...                       self._hash_action(action),
            ...                       reward, done, 
            ...                       self._hash_observation(next_observation), 
            ...                       info, **param)
        """
        # self.memory.remember(self._hash_observation(observation), self._hash_action(action), reward, done, self._hash_observation(next_observation), info, **param)
        raise NotImplementedError
    
    def forget(self):
        """ Make the agent forget his memory """
        self.memory.forget()
    


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

    def turn(self, state):
        """Give the turn to the next agent to play
        
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


class Playground():

    """ A playground is used to run agent(s) on an environement.

    Attributes
    ----------
        env: :class:`MultiEnv` or |gym.Env|
            The environement in which agents will play
        agents: :class:`list`
            The list of :class:`Agent` in the :class:`Playground`
    """

    def __init__(self, environement:Env, agents):
        assert isinstance(environement, Env)
        if isinstance(agents, Agent):
            agents = [agents]
        for agent in agents:
            assert isinstance(agent, Agent)
        
        self.env = environement
        self.agents = agents

    def run(self, episodes, render=True, learn=True, verbose=0):
        """Let the agent(s) play on the environement for a number of episodes."""
        print_cycle = max(1, episodes // 100)
        avg_gain = np.zeros_like(self.agents)
        steps = 0
        t0 = time()
        for episode in range(1, episodes+1):

            observation = self.env.reset()
            previous = np.array([{'observation':None, 'action':None, 'reward':None, 'done':None, 'info':None}]*len(self.agents))
            done = False
            gain = np.zeros_like(avg_gain)
            step = 0

            while not done:

                if render: self.env.render()

                if isinstance(self.env, MultiEnv):
                    agent_id = self.env.turn(observation)
                else: agent_id = 0

                prev = previous[agent_id]
                if learn and prev['observation'] is not None:
                    agent.remember(prev['observation'], prev['action'], prev['reward'], prev['done'], observation, prev['info'])
                    agent.learn()
                
                agent = self.agents[agent_id]
                action = agent.act(observation)
                next_observation, reward, done , info = self.env.step(action)
                gain[agent_id] += reward
                step += 1

                if learn:
                    for key, value in zip(prev, [observation, action, reward, done, info]):
                        prev[key] = value
                    if done:
                        agent.remember(prev['observation'], prev['action'], prev['reward'], prev['done'], observation, prev['info'])
                        agent.learn()

                if verbose > 1:
                    print(f"------ Step {step} ------ Player is {agent_id}\nobservation:\n{observation}\naction:\n{action}\nreward:{reward}\ndone:{done}\nnext_observation:\n{next_observation}\ninfo:{info}")
                observation = next_observation
            
            if verbose > 0:
                steps += step
                avg_gain += gain
                if episode%print_cycle==0: 
                    print(f"Episode {episode}/{episodes}    \t gain({print_cycle}):{avg_gain/print_cycle} \t"
                          f"explorations:{np.array([agent.control.exploration for agent in self.agents])}\t"
                          f"steps/s:{steps/(time()-t0):.0f}, episodes/s:{print_cycle/(time()-t0):.0f}")
                    avg_gain = np.zeros_like(self.agents)
                    steps = 0
                    t0 = time()

    def fit(self, episodes, verbose=0):
        """Train the agent(s) on the environement for a number of episodes."""
        self.run(episodes, render=False, learn=True, verbose=verbose)

    def test(self, episodes, verbose=0):
        """Test the agent(s) on the environement for a number of episodes."""
        self.run(episodes, render=True, learn=False, verbose=verbose)


