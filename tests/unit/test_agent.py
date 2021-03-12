# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=protected-access, attribute-defined-outside-init, unused-argument

""" Test agent.py """

import pytest
import pytest_check as check

from learnrl.agent import Agent

class TestAgent:

    """Agent"""

    def test_init(self):
        """ should instanciate correctly. """
        Agent()

    def test_act(self):
        """ act should raise NotImplementedError. """
        agent = Agent()
        with pytest.raises(NotImplementedError):
            agent.act('obs')

    def test_learn(self):
        """ learn should return an empty dictionary by default. """
        agent = Agent()
        logs = agent.learn()
        check.equal(logs, {})

    def test_remember(self):
        """ remember should pass by default. """
        agent = Agent()
        agent.remember(
            'observation',
            'action',
            'reward',
            'done',
            next_observation='next_observation',
            info={'info': 'info'},
        )
