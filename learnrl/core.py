# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import numpy as np
import collections.abc as collections

from time import time
from gym import Env


class Memory():

    MEMORY_KEYS = ('state', 'action', 'reward', 'done', 'next_state', 'info')
    max_memory_len = 1e4

    def __init__(self):
        self.datas = {key:None for key in self.MEMORY_KEYS}

    def remember(self, state, action, reward, done, next_state=None, info={}, **param):
        for key, value in zip(self.MEMORY_KEYS, (state, action, reward, done, next_state, info)):

            # Check that value is an instance of numpy.ndarray or transform the value
            if isinstance(value, collections.Sequence) or type(value) != np.ndarray or value.ndim < 1:
                value = np.array([value])

            # Add the new experience into the memory forgetting long past experience if neccesary
            if self.datas[key] is None:
                self.datas[key] = value
            else:
                if len(self.datas[key]) < self.max_memory_len:
                    self.datas[key] = np.concatenate((self.datas[key], value), axis=0)
                else:
                    self.datas[key] = np.roll(self.datas[key], shift=-1, axis=0)
                    self.datas[key][-1] = np.array(value)
        
        # Same for suplementary parameters
        for key in param:
            if self.datas[key] is None:
                self.datas[key] = param[key]
            else:
                if len(self.datas[key]) < self.max_memory_len:
                    self.datas[key] = np.concatenate((self.datas[key], param[key]), axis=0)
                else:
                    self.datas[key] = np.roll(self.datas[key], shift=-1, axis=0)
                    self.datas[key][-1] = np.array(param[key])

    
    def forget(self):
        self.datas = {key:None for key in self.MEMORY_KEYS}


class Agent():

    name = 'DefaultAgent'

    state_values = None
    state_visits = None

    action_values = None
    action_visits = None
    
    memory = Memory()

    def policy(self, observation, legal_actions):
        raise NotImplementedError

    def act(self, observation, legal_actions):
        raise NotImplementedError
    
    @staticmethod
    def _hash_state(state):
        return state

    @staticmethod
    def _hash_action(action):
        return action

    def remember(self, state, action, reward, done, next_state=None, info={}, **param):
        self.memory.remember(self._hash_state(state), self._hash_action(action), reward, done, self._hash_state(next_state), info, **param)
    
    def forget(self):
        self.memory.forget()
    
    def learn(self):
        raise NotImplementedError
    
    def render(self):
        raise NotImplementedError


class MultiEnv(Env):

    r"""
    A layer over the Gym Env class able to handle environements with multiple agents.
    
    The main add in MultiEnv is the method:
        turn: returns the next agent to play given the state

    On top of the main API basic methodes:
        step: take a step of the environement given the action of the active player
        reset  
        render  
        close  
        seed

    And basic attributes:

        action_space: The Space object corresponding to valid actions  
        observation_space: The Space object corresponding to valid observations  
        reward_range: A tuple corresponding to the min and max possible rewards  

    Note: a default reward range set to [-inf,+inf] already exists. Set it if you want a narrower range.
    """

    def turn(self, state):
        raise NotImplementedError 


class Playground():

    """
    A playground is used to train and test one or multiple agents on an environement.

    Typical use involve the methodes :
        run : main method
        fit : to train the agent(s) without rendering
        test : to test the agent(s) without learning
    
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
        np.set_printoptions(precision=3)
        print_cycle = max(1, episodes // 100)
        avg_gain = np.zeros_like(self.agents)
        steps = 0
        t0 = time()
        for episode in range(episodes):

            state = self.env.reset()
            previous = np.array([{'state':None, 'action':None, 'reward':None, 'done':None, 'info':None}]*len(self.agents))
            done = False
            gain = np.zeros_like(avg_gain)
            step = 0

            while not done:

                if render: self.env.render()

                if isinstance(self.env, MultiEnv):
                    agent_id = self.env.turn(state)
                else: agent_id = 0

                prev = previous[agent_id]
                if learn and prev['state'] is not None:
                    agent.remember(prev['state'], prev['action'], prev['reward'], prev['done'], state, prev['info'])
                    agent.learn()
                
                agent = self.agents[agent_id]
                action = agent.act(state)
                next_state, reward, done , info = self.env.step(action)
                gain[agent_id] += reward
                step += 1

                if learn:
                    for key, value in zip(prev, [state, action, reward, done, info]):
                        prev[key] = value

                if verbose > 1:
                    print(f"------ Step {step} ------ Player is {agent_id}\nobservation:\n{state}\naction:\n{action}\nreward:{reward}\ndone:{done}\nnext_observation:\n{next_state}\ninfo:{info}")
                state = next_state
            
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


