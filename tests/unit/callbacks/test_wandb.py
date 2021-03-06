""" Test wandb callback """

# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import importlib
import pytest

import learnrl

wandb_spec = importlib.util.find_spec('wandb')

if wandb_spec is not None:
    def test_instanciate_without_wandb(hide_wandb):
        """ should raise an ImportError when wandb is not installed. """
        with pytest.raises(ImportError, match=r".*wandb >= 0.10.*"):
            learnrl.callbacks.WandbCallback()

    def test_instanciate_with_wandb():
        """ should load normally when wandb is installed """
        learnrl.callbacks.WandbCallback()
else:
    pass
