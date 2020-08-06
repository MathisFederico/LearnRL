# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
import numpy as np
from learnrl.control import Control, Greedy, Ucb, Puct
from learnrl.estimators import TableEstimator
from gym.spaces import Discrete

@pytest.fixture
def action_values():
    action_values = TableEstimator(Discrete(2), Discrete(3))
    action_values.load(np.array([[1, 0, -1], [-1, 1, 0]]))
    return action_values

@pytest.fixture
def action_visits():
    action_visits = TableEstimator(Discrete(2), Discrete(3))
    action_visits.load(np.array([[100, 8, 3], [3, 100, 8]], dtype=np.uint8))
    return action_visits

@pytest.fixture
def observations():
    return np.array([0, 1, 0])

# Test general control features
def test_name():
    with pytest.raises(ValueError):
        Control()

def test_update_exploration():
    initial_exploration = 0.378
    exploration_decay = 0.9
    control = Control(name='default', exploration=initial_exploration, exploration_decay=exploration_decay)
    if control.exploration != initial_exploration:
        raise ValueError(f"Exploration is {control.exploration} instead of {initial_exploration}")
    
    exploration = 1.0
    control.update_exploration(exploration=exploration)
    if control.exploration != exploration:
        raise ValueError(f"Exploration is {control.exploration} instead of fixed updated exploration {exploration}")
    
    control.update_exploration()
    expected_exploration = exploration*(1-exploration_decay)
    if control.exploration != expected_exploration:
        raise ValueError(f"Exploration is {control.exploration} and did not decay to {expected_exploration}")

def test_get_policy(action_values):
    control = Control(name="dummy_control")
    with pytest.raises(NotImplementedError):
        control._get_policy(observations=[0], action_values=action_values)

# Test classic controls

def test_Greedy(action_values, action_visits, observations):
    expected_policies = {}
    expected_policies[0] = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [1, 0, 0]])

    expected_policies[0.5] = np.array([[0.5 + 0.5/3, 0.5/3, 0.5/3],
                                       [0.5/3, 0.5 + 0.5/3, 0.5/3],
                                       [0.5 + 0.5/3, 0.5/3, 0.5/3]])
    
    expected_policies[1] = np.array([[1/3, 1/3, 1/3],
                                     [1/3, 1/3, 1/3],
                                     [1/3, 1/3, 1/3]])

    for eps in [0, 0.5, 1]:
        greedy = Greedy(eps)
        policy = greedy._get_policy(observations, action_values=action_values)
        expected_policy = expected_policies[eps]
        if not np.all(policy==expected_policy):
            raise ValueError(f'Wrong policy for greedy with eps={eps}\
                                \nExpected {expected_policy}\
                                \nGot {policy}')

def test_vectorized_Greedy(action_values, action_visits, observations):
    expected_policies = np.array([[0.5 + 0.5/3, 0.5/3, 0.5/3],
                                 [0.5/3, 0.5 + 0.5/3, 0.5/3]])

    greedy = Greedy(0.5)
    policy = greedy._get_policy(np.array([0, 1]), action_values=action_values)
    expected_policy = expected_policies
    if not np.all(policy==expected_policy):
        raise ValueError(f'Wrong policy for greedy with eps={0.5}\
                            \nExpected {expected_policy}\
                            \nGot {policy}')

def test_Ucb(action_values, action_visits, observations):
    ucb = Ucb()
    with pytest.raises(ValueError):
        ucb._get_policy(observations, action_values=None, action_visits=None)

    cs = [0, 2, 5]
    expected_policies = {}
    for i, c in enumerate(cs):
        temp = np.zeros((3, 3))
        temp[0, i] = 1
        temp[1, (i+1)%3] = 1
        temp[2, i] = 1
        expected_policies[c] = temp

    for c in cs:
        ucb = Ucb(exploration=c)
        policy = ucb._get_policy(observations, action_values=action_values, action_visits=action_visits)
        expected_policy = expected_policies[c]
        if not np.all(policy==expected_policy):
            raise ValueError(f'Wrong policy for Ucb with c={c}\
                                \nExpected {expected_policy}\
                                \nGot {policy}')

def test_Puct(action_values, action_visits, observations):
    puct = Puct(10)

    with pytest.raises(ValueError):
        puct._get_policy(observations, action_values=None, action_visits=None)

    cs = [0, 0.5, 1]
    expected_policies = {}
    for i, c in enumerate(cs):
        temp = np.zeros((3, 3))
        temp[0, i] = 1
        temp[1, (i+1)%3] = 1
        temp[2, i] = 1
        expected_policies[c] = temp

    for c in cs:
        ucb = Puct(exploration=c)
        policy = ucb._get_policy(observations, action_values=action_values, action_visits=action_visits)
        expected_policy = expected_policies[c]
        if not np.all(policy==expected_policy):
            raise ValueError(f'Wrong policy for PUCT with c={c}\
                                \nExpected {expected_policy}\
                                \nGot {policy}')

