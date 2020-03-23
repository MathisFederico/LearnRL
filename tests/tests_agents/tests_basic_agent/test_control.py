# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
import numpy as np
from learnrl.agents.table.control import Control, Greedy, UCB, Puct

# Test basic control features
def test_name():
    with pytest.raises(ValueError):
        Control(4)

def test_update_exploration():
    initial_exploration = 0.378
    exploration_decay = 0.9
    control = Control(4, name='default', initial_exploration=initial_exploration, exploration_decay=exploration_decay)
    if control.exploration != initial_exploration:
        raise ValueError(f"Learning rate is {control.exploration} instead of initial_learning rate {initial_exploration}")
    
    exploration = 1.0
    control.update_exploration(exploration=exploration)
    if control.exploration != exploration:
        raise ValueError(f"Learning rate is {control.exploration} instead of fixed updated learning rate {exploration}")
    
    control.update_exploration()
    if control.exploration != exploration*exploration_decay:
        raise ValueError(f"Learning rate is {control.exploration} and did not decay to {exploration*exploration_decay}")

def test_get_policy():
    control = Control(3, name="default_policy")
    with pytest.raises(NotImplementedError):
        control.get_policy(state=0, action_values=None)
    
def test_check_policy():
    raise NotImplementedError

# Test classic controls
@pytest.fixture
def action_values():
    return np.array([[1, 0, -1], [-1, 1, 0]])

@pytest.fixture
def action_visits():
    return np.array([[100, 8, 3], [3, 100, 8]], dtype=np.uint16)

def test_Greedy(action_values, action_visits):
    expected_policies = {}
    expected_policies[0] = np.array([1, 0, 0])
    expected_policies[0.5] = np.array([0.5 + 0.5/3, 0.5/3, 0.5/3])
    expected_policies[1] = np.array([1/3, 1/3, 1/3])

    for eps in [0, 0.5, 1]:
        greedy = Greedy(3, eps)
        policy = greedy.get_policy(state=0, action_values=action_values)
        expected_policy = expected_policies[eps]
        if not np.all(policy==expected_policy):
            raise ValueError(f'Wrong policy for greedy with eps={eps}\
                                \nExpected {expected_policy}\
                                \nGot {policy}')

def test_vectorized_Greedy(action_values, action_visits):
    expected_policies = np.array([[0.5 + 0.5/3, 0.5/3, 0.5/3],
                                 [0.5/3, 0.5 + 0.5/3, 0.5/3]])

    greedy = Greedy(3, 0.5)
    policy = greedy.get_policy(state=np.array([0, 1]), action_values=action_values)
    expected_policy = expected_policies
    if not np.all(policy==expected_policy):
        raise ValueError(f'Wrong policy for greedy with eps={0.5}\
                            \nExpected {expected_policy}\
                            \nGot {policy}')

def test_UCB(action_values, action_visits):
    ucb = UCB(3)
    with pytest.raises(ValueError):
        ucb.get_policy(state=0, action_values=None, action_visits=None)

    cs = [0, 2, 5]
    expected_policies = {}
    for i, c in enumerate(cs):
        temp = np.zeros(3)
        temp[i] = 1
        expected_policies[c] = temp

    for c in cs:
        ucb = UCB(3, initial_exploration=c)
        policy = ucb.get_policy(state=0, action_values=action_values, action_visits=action_visits)
        expected_policy = expected_policies[c]
        if not np.all(policy==expected_policy):
            raise ValueError(f'Wrong policy for UCB with c={c}\
                                \nExpected {expected_policy}\
                                \nGot {policy}')

def test_Puct(action_values, action_visits):
    puct = Puct(10)

    with pytest.raises(ValueError):
        puct.get_policy(state=0, action_values=None, action_visits=None)

    cs = [0, 0.5, 1]
    expected_policies = {}
    for i, c in enumerate(cs):
        temp = np.zeros(3)
        temp[i] = 1
        expected_policies[c] = temp

    for c in cs:
        ucb = Puct(3, initial_exploration=c)
        policy = ucb.get_policy(state=0, action_values=action_values, action_visits=action_visits)
        expected_policy = expected_policies[c]
        if not np.all(policy==expected_policy):
            raise ValueError(f'Wrong policy for UCB with c={c}\
                                \nExpected {expected_policy}\
                                \nGot {policy}')

