import pytest
import numpy as np

class ControlError(Exception):
    pass

@pytest.fixture
def action_values():
    return np.array(range(10)[::-1])

@pytest.fixture
def action_visits():
    return np.array(range(10)[::-1])

def test_getPolicy():
    from agents.basic.control import Control
    control = Control(name="default_policy")
    with pytest.raises(NotImplementedError):
        control.getPolicy(action_values=None)

def test_Greedy(action_values, action_visits):
    from agents.basic.control import Greedy

    expected_policies = {}
    expected_policies[0] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    expected_policies[0.5] = np.array([0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
    expected_policies[1] = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    for eps in [0, 0.5, 1]:
        greedy = Greedy(eps)
        policy = greedy.getPolicy(action_values=action_values)
        expected_policy = expected_policies[eps]
        if not np.all(policy==expected_policy):
            raise ControlError(f'Wrong policy for greedy with eps={eps}\
                                \nExpected {expected_policy}\
                                \nGot {policy}')

def test_UCB(action_values, action_visits):
    from agents.basic.control import UCB

    ucb = UCB()
    with pytest.raises(ValueError):
        ucb.getPolicy(action_values=None, action_visits=None)

    expected_policies = {}
    for c in [1, 2, 10]:
        if c < 5:
            expected_policies[c] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            expected_policies[c] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    for c in [1, 2, 10]:
        ucb = UCB(initial_exploration=c)
        policy = ucb.getPolicy(action_values=action_values, action_visits=action_visits)
        expected_policy = expected_policies[c]
        if not np.all(policy==expected_policy):
            raise ControlError(f'Wrong policy for UCB with c={c}\
                                \nExpected {expected_policy}\
                                \nGot {policy}')

def test_Puct(action_values, action_visits):
    from agents.basic.control import Puct
    puct = Puct()

    with pytest.raises(ValueError):
        puct.getPolicy(action_values=None, action_visits=None)

    expected_policies = {}
    for c in [1, 2, 10]:
        if c < 2:
            expected_policies[c] = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            expected_policies[c] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

    for c in [1, 2, 10]:
        ucb = Puct(initial_exploration=c)
        policy = ucb.getPolicy(action_values=action_values, action_visits=action_visits)
        expected_policy = expected_policies[c]
        if not np.all(policy==expected_policy):
            raise ControlError(f'Wrong policy for UCB with c={c}\
                                \nExpected {expected_policy}\
                                \nGot {policy}')