# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
import numpy as np
np.random.seed(42)

from copy import deepcopy

from learnrl.environments import RdNimEnv
from learnrl.agents import TableAgent
from learnrl.core import Memory
from learnrl.agents.table.evaluation import MonteCarlo, TemporalDifference, QLearning
from learnrl.agents.table.control import Greedy

@pytest.fixture
def memory():
    memory = Memory()
    memory.remember(0, 0, 0, False, 1)
    memory.remember(1, 0, -1, True, 2)
    memory.legal_actions = {0:[0, 1], 1:[0, 1]}
    return memory

@pytest.fixture
def action_values():
    return {(0,0):1, (0,1):0, (1, 0):1}

@pytest.fixture
def action_visits():
    return {(0,0):1, (0,1):1, (1, 0):1}

# Monte Carlo
@pytest.mark.slow
def test_nim_optimal_policy():
    n_triplets = 2
    n_sticks = n_triplets*4 + 2
    env = RdNimEnv(initial_state=n_sticks, is_optimal=True)
    agent = TableAgent(observation_space=env.observation_space, action_space=env.action_space,
                       evaluation=MonteCarlo(initial_learning_rate=0.3),
                       control=Greedy(env.action_space.n, initial_exploration=0))

    n_games = 50
    legal_actions = np.array(range(3))

    for _ in range(n_games):
        done = False
        observation = env.reset()
        while not done:
            action = agent(observation, legal_actions)
            next_observation, reward, done , info = env.step(action)
            agent.remember(observation, action, reward, done, next_observation, info)
            agent.learn()
            observation = deepcopy(next_observation)
    
    action_size, observation_size = env.action_space.n, env.observation_space.n
    action_values = np.zeros((action_size, observation_size))
    for action in range(action_size):
        for observation in range(observation_size):
            try:
                action_values[action, observation] = agent.action_values[(observation, action)]
            except KeyError:
                pass
    
    greedy_actions = 1+np.argmax(action_values, axis=0)
    pertinent_observations = np.concatenate([[2+k+4*i for k in range(3)] for i in range(n_triplets)])
    pertinent_actions = greedy_actions[pertinent_observations]
    expected_actions = np.concatenate([(1, 2, 3) for _ in range(n_triplets)])
    if not np.all(pertinent_actions==expected_actions):
        raise ValueError(f'MonteCarlo agent got policy {pertinent_actions} instead of {expected_actions}')


# TemporalDifference - online
@pytest.mark.slow
def test_td_onl_onp_nim_optimal_policy():
    n_triplets = 2
    n_sticks = n_triplets*4 + 2
    env = RdNimEnv(initial_state=n_sticks, is_optimal=True)
    agent = TableAgent(observation_space=env.observation_space, action_space=env.action_space,
                       evaluation=TemporalDifference(initial_learning_rate=0.3),
                       control=Greedy(env.action_space.n, initial_exploration=0))

    n_games = 30
    legal_actions = np.array(range(3))

    for _ in range(n_games):
        done = False
        observation = env.reset()
        while not done:
            action = agent(observation, legal_actions)
            next_observation, reward, done , info = env.step(action)
            agent.remember(observation, action, reward, done, next_observation, info)
            agent.learn()
            observation = deepcopy(next_observation)
    
    action_size, observation_size = env.action_space.n, env.observation_space.n
    action_values = np.zeros((action_size,observation_size))
    for action in range(action_size):
        for observation in range(observation_size):
            try:
                action_values[action, observation] = agent.action_values[(observation, action)]
            except KeyError:
                pass
    
    greedy_actions = 1+np.argmax(action_values, axis=0)
    pertinent_observations = np.concatenate([[2+k+4*i for k in range(3)] for i in range(n_triplets)])
    pertinent_actions = greedy_actions[pertinent_observations]
    expected_actions = np.concatenate([(1, 2, 3) for _ in range(n_triplets)])
    if not np.all(pertinent_actions==expected_actions):
        raise ValueError(f'TD-Onl-Onp agent got policy {pertinent_actions} instead of {expected_actions}')


# TemporalDifference - offline
@pytest.mark.slow
def test_td_offl_onp_nim_optimal_policy():
    n_triplets = 2
    n_sticks = n_triplets*4 + 2
    env = RdNimEnv(initial_state=n_sticks, is_optimal=True)
    agent = TableAgent(observation_space=env.observation_space, action_space=env.action_space,
                       evaluation=TemporalDifference(initial_learning_rate=0.3),
                       control=Greedy(env.action_space.n, initial_exploration=0))

    n_games = 30
    legal_actions = np.array(range(3))

    for _ in range(n_games):
        done = False
        observation = env.reset()
        while not done:
            action = agent(observation, legal_actions)
            next_observation, reward, done , info = env.step(action)
            agent.remember(observation, action, reward, done, next_observation, info)
            agent.learn(online=False)
            observation = deepcopy(next_observation)
    
    action_size, observation_size = env.action_space.n, env.observation_space.n
    action_values = np.zeros((action_size,observation_size))
    for action in range(action_size):
        for observation in range(observation_size):
            try:
                action_values[action, observation] = agent.action_values[(observation, action)]
            except KeyError:
                pass
    
    greedy_actions = 1+np.argmax(action_values, axis=0)
    pertinent_observations = np.concatenate([[2+k+4*i for k in range(3)] for i in range(n_triplets)])
    pertinent_actions = greedy_actions[pertinent_observations]
    expected_actions = np.concatenate([(1, 2, 3) for _ in range(n_triplets)])
    if not np.all(pertinent_actions==expected_actions):
        raise ValueError(f'TD-Onl-Onp agent got policy {pertinent_actions} instead of {expected_actions}')