import pytest

from agents import BasicAgent
import numpy as np
from gym.spaces import Discrete

@pytest.fixture
def agent():
    return BasicAgent(Discrete(3), Discrete(4))

def test_name(agent):
    default_name = r"basic_greedy_mc_{}"
    if agent.name != default_name:
        raise ValueError(f'Default name should be {default_name} and not {agent.name}')
    if str(agent) != default_name:
        raise ValueError(f'str(BasicAgent) should be {default_name}(agent name) and not {str(agent)}')
    
    agent = BasicAgent(Discrete(3), Discrete(4), foo='foo')
    default_name = r"basic_greedy_mc_{'foo': 'foo'}"
    if agent.name != default_name:
        raise ValueError(f'Default name should be {default_name} and not {agent.name}')

def test_hash(agent):
    raise NotImplementedError

def test_act(agent):
    raise NotImplementedError