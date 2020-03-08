import pytest
import numpy as np

from agents.basic.evaluation import Evaluation, MonteCarlo, TemporalDifference
from agents.agent import Memory

# Test basic evaluation features
def test_name():
    with pytest.raises(ValueError):
        Evaluation()

def test_str():
    name = 'default'
    evaluation = Evaluation(name=name)
    if str(evaluation) != name:
        raise ValueError(f"str(Evaluation) should return {name} and not {str(evaluation)}")

def test_update_learning_rate():
    initial_learning_rate = 0.378
    learning_rate_decay = 0.9
    evaluation = Evaluation(name='default', initial_learning_rate=initial_learning_rate, learning_rate_decay=learning_rate_decay)
    if evaluation.learning_rate != initial_learning_rate:
        raise ValueError(f"Learning rate is {evaluation.learning_rate} instead of initial_learning rate {initial_learning_rate}")
    
    learning_rate = 1.0
    evaluation.update_learning_rate(learning_rate=learning_rate)
    if evaluation.learning_rate != learning_rate:
        raise ValueError(f"Learning rate is {evaluation.learning_rate} instead of fixed updated learning rate {learning_rate}")
    
    evaluation.update_learning_rate()
    if evaluation.learning_rate != learning_rate*learning_rate_decay:
        raise ValueError(f"Learning rate is {evaluation.learning_rate} and did not decay to {learning_rate*learning_rate_decay}")

# Test learning of classic evaluations
@pytest.fixture
def memory():
    memory = Memory()
    return memory

@pytest.fixture
def action_values():
    return np.array([[1, 0, -1], [-1, 1, 0]])

@pytest.fixture
def action_visits():
    return np.array([[100, 8, 3], [3, 100, 8]], dtype=np.uint16)

def test_monte_carlo(action_values, action_visits, memory):
    evaluation = MonteCarlo(initial_learning_rate=0.1)
    raise NotImplementedError

def test_td_0_onl_onp(action_values, action_visits, memory):
    evaluation = TemporalDifference(initial_learning_rate=0.1, online=True)
    raise NotImplementedError

def test_td_0_offl_offp(action_values, action_visits, memory):
    evaluation = TemporalDifference(initial_learning_rate=0.1, online=False, target_policy=None)
    raise NotImplementedError

def test_td_lamb_offl_offp(action_values, action_visits, memory):
    evaluation = TemporalDifference(lamb=0.8, initial_learning_rate=0.1, online=False, target_policy=None)
    raise NotImplementedError
