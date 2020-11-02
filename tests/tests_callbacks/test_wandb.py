# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
import learnrl
import importlib
wandb_spec = importlib.util.find_spec('wandb')

if wandb_spec is not None:
    def test_instanciate_without_tensorflow(hide_wandb):
        with pytest.raises(ImportError, match=r".*wandb >= 0.10.*"):
            learnrl.callbacks.WandbLogger()

    def test_instanciate_with_tensorflow():
        learnrl.callbacks.WandbLogger()
else:
    pass
