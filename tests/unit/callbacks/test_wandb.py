# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
import learnrl
import importlib
wandb_spec = importlib.util.find_spec('wandb')

if wandb_spec is not None:

    class TestWandbCallback():

        def test_instanciate_without_wandb(self, hide_wandb): # pylint: disable=unused-argument
            """Should raise ImportError if wandb is not found"""
            with pytest.raises(ImportError, match=r".*wandb >= 0.10.*"):
                learnrl.callbacks.WandbCallback()

        def test_instanciate_with_wandb(self):
            """Should instanciate correctly if wandb is found"""
            learnrl.callbacks.WandbCallback()
else:
    pass
