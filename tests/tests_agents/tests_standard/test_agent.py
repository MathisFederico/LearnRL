# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest

from learnrl.agents import RandomAgent, StandardAgent
from learnrl.control import Greedy
import numpy as np
from gym.spaces import Discrete, MultiDiscrete

def test_name():
    agent = StandardAgent(Discrete(3), Discrete(4))
    default_name = r"standard_greedy_qlearning_{}"
    if agent.name != default_name:
        raise ValueError(f'Default name should be {default_name} and not {agent.name}')
    if str(agent) != default_name:
        raise ValueError(f'str(BasicAgent) should be {default_name}(agent name) and not {str(agent)}')
    
    agent = StandardAgent(Discrete(3), Discrete(4), foo='foo')
    default_name = r"standard_greedy_qlearning_{'foo': 'foo'}"
    if agent.name != default_name:
        raise ValueError(f'Default name should be {default_name} and not {agent.name}')

