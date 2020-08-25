# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
import numpy as np

from learnrl.evaluation import Evaluation, MonteCarlo, TemporalDifference
from learnrl.memory import Memory

# Test basic evaluation features
def test_name():
    with pytest.raises(ValueError):
        Evaluation()

def test_str():
    name = 'default'
    evaluation = Evaluation(name=name)
    if str(evaluation) != name:
        raise ValueError(f"str(Evaluation) should return {name} and not {str(evaluation)}")


# # Test learning of classic evaluations
# @pytest.fixture
# def memory():
#     memory = Memory()
#     return memory

# @pytest.fixture
# def action_values():
#     return np.array([[1, 0, -1], [-1, 1, 0]])

# @pytest.fixture
# def action_visits():
#     return np.array([[100, 8, 3], [3, 100, 8]], dtype=np.uint16)

