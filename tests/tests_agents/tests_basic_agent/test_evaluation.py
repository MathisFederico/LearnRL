# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
import numpy as np

from learnrl.agents.table.evaluation import Evaluation, MonteCarlo, TemporalDifference
from learnrl.core import Memory

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

