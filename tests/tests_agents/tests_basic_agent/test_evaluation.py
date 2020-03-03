import pytest
import numpy as np
from copy import deepcopy

@pytest.fixture
def memory():
    from agents import Memory
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
    from envs import NimEnv
    from agents import BasicAgent
    from agents.basic.evaluation import MonteCarlo

    env = NimEnv(is_optimal=True)
    agent = BasicAgent(evaluation=MonteCarlo())

    n_games = 2000
    legal_actions = np.array(range(3))

    for _ in range(n_games):
        done = False
        state = env.reset()
        while not done:
            action = agent.act(state, legal_actions)
            next_state, reward, done , info = env.step(action)
            agent.remember(state, action, reward, done, next_state, info)
            agent.learn()
            state = deepcopy(next_state)
    
    action_size, state_size = env.action_space.n, env.observation_space.n
    action_values = np.zeros((action_size,state_size))
    for action in range(action_size):
        for state in range(state_size):
            try:
                action_values[action, state] = agent.action_values[(state, action)]
            except KeyError:
                pass
    
    greedy_actions = 1+np.argmax(action_values, axis=0)
    pertinent_states = np.concatenate([[2+k+4*i for k in range(3)] for i in range(4)])
    pertinent_actions = greedy_actions[pertinent_states]
    expected_actions = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    if not np.all(pertinent_actions==expected_actions):
        raise ValueError(f'MonteCarlo agent got policy {pertinent_actions} instead of {expected_actions}')


# TemporalDifference
@pytest.mark.slow
def test_td_onl_onp_nim_optimal_policy():
    from envs import NimEnv
    from agents import BasicAgent
    from agents.basic.evaluation import TemporalDifference

    env = NimEnv(is_optimal=True)
    agent = BasicAgent(evaluation=TemporalDifference())

    n_games = 2000
    legal_actions = np.array(range(3))

    for _ in range(n_games):
        done = False
        state = env.reset()
        while not done:
            action = agent.act(state, legal_actions)
            next_state, reward, done , info = env.step(action)
            agent.remember(state, action, reward, done, next_state, info)
            agent.learn()
            state = deepcopy(next_state)
    
    action_size, state_size = env.action_space.n, env.observation_space.n
    action_values = np.zeros((action_size,state_size))
    for action in range(action_size):
        for state in range(state_size):
            try:
                action_values[action, state] = agent.action_values[(state, action)]
            except KeyError:
                pass
    
    greedy_actions = 1+np.argmax(action_values, axis=0)
    pertinent_states = np.concatenate([[2+k+4*i for k in range(3)] for i in range(4)])
    pertinent_actions = greedy_actions[pertinent_states]
    expected_actions = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
    if not np.all(pertinent_actions==expected_actions):
        raise ValueError(f'TD-Onl-Onp agent got policy {pertinent_actions} instead of {expected_actions}')
