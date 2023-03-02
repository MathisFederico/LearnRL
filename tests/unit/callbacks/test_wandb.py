# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=unused-argument

""" Test WandbCallback """

import importlib.util
import pytest

import benchmarks

wandb_spec = importlib.util.find_spec('wandb')

if wandb_spec is not None:
    class TestWandbCallback():
        """ WandbCallback """

        def test_instanciate_without_wandb(self, hide_wandb):
            """ should raise an ImportError when wandb is not found. """
            with pytest.raises(ImportError, match=r".*wandb >= 0.10.*"):
                benchmarks.callbacks.WandbCallback(run=None)

        def test_instanciate_with_wandb(self):
            """ should instanciate correctly if wandb is found. """
            benchmarks.callbacks.WandbCallback(run=None)
else:
    pass
