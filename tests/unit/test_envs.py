# pylint: disable=protected-access, attribute-defined-outside-init, unused-argument

""" Test agent.py """

import pytest

from benchmarks.envs import TurnEnv


class TestTurnEnv:

    """TurnEnv"""

    def test_step(self):
        """step should raise NotImplementedError."""
        env = TurnEnv()
        with pytest.raises(NotImplementedError):
            env.step("action")

    def test_turn(self):
        """turn should raise NotImplementedError."""
        env = TurnEnv()
        with pytest.raises(NotImplementedError):
            env.turn("observation")

    def test_reset(self):
        """reset should raise NotImplementedError."""
        env = TurnEnv()
        with pytest.raises(NotImplementedError):
            env.reset()
