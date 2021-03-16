# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=protected-access, attribute-defined-outside-init, unused-argument

""" Test agent.py """

import pytest

from learnrl.envs import TurnEnv


class TestTurnEnv:

    """ TurnEnv """

    def test_step(self):
        """ step should raise NotImplementedError. """
        env = TurnEnv()
        with pytest.raises(NotImplementedError):
            env.step('action')

    def test_turn(self):
        """ turn should raise NotImplementedError. """
        env = TurnEnv()
        with pytest.raises(NotImplementedError):
            env.turn('observation')

    def test_reset(self):
        """ reset should raise NotImplementedError. """
        env = TurnEnv()
        with pytest.raises(NotImplementedError):
            env.reset()
