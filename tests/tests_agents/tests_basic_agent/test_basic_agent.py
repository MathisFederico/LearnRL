import pytest

from agents import BasicAgent, QLearningAgent
from agents.basic.control import Greedy
import numpy as np
from gym.spaces import Discrete, MultiDiscrete

@pytest.fixture
def agent():
    return BasicAgent(Discrete(3), Discrete(4))

@pytest.fixture
def qlearning_agent():
    return QLearningAgent(Discrete(3), Discrete(4))

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

def test_name_qlearning(qlearning_agent):
    agent = qlearning_agent
    default_name = r"qlearning_greedy_{'online': True}"
    if agent.name != default_name:
        raise ValueError(f'Default name should be {default_name} and not {agent.name}')
    
    agent = QLearningAgent(Discrete(3), Discrete(4), foo='foo')
    default_name = r"qlearning_greedy_{'foo': 'foo', 'online': True}"
    if agent.name != default_name:
        raise ValueError(f'Default name should be {default_name} and not {agent.name}')

def test_qlearning_evaluation(qlearning_agent):
    evaluation = qlearning_agent.evaluation
    if evaluation.name != 'td':
        raise ValueError(f"QLearningAgent evaluation should be 'td' and not {evaluation.name}")
    default_control = Greedy(4, initial_exploration=0)
    if evaluation.target_control != default_control:
        raise ValueError(f"QLearningAgent target_policy should be {default_control} and not {evaluation.target_control}")

def test_qlearning_prevent_params():
    with pytest.raises(ValueError):
        QLearningAgent(Discrete(3), Discrete(4), evaluation='foo')
    with pytest.raises(ValueError):
        QLearningAgent(Discrete(3), Discrete(4), target_control=Greedy(4, initial_exploration=1))
    with pytest.raises(ValueError):
        QLearningAgent(Discrete(3), Discrete(4), online=False)

def test_hash_discrete(agent):
    raise NotImplementedError

def test_hash_multidiscrete():
    agent = BasicAgent(MultiDiscrete((2, 3)), MultiDiscrete((4, 2)))
    raise NotImplementedError

def test_act(agent):
    raise NotImplementedError
