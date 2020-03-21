# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest

from learnrl.agents import TableAgent
from learnrl.agents.table.control import Greedy
import numpy as np
from gym.spaces import Discrete, MultiDiscrete

@pytest.fixture
def agent():
    return TableAgent(Discrete(3), Discrete(4))

def test_name(agent):
    default_name = r"table_greedy_qlearning_{}"
    if agent.name != default_name:
        raise ValueError(f'Default name should be {default_name} and not {agent.name}')
    if str(agent) != default_name:
        raise ValueError(f'str(BasicAgent) should be {default_name}(agent name) and not {str(agent)}')
    
    agent = TableAgent(Discrete(3), Discrete(4), foo='foo')
    default_name = r"table_greedy_qlearning_{'foo': 'foo'}"
    if agent.name != default_name:
        raise ValueError(f'Default name should be {default_name} and not {agent.name}')

