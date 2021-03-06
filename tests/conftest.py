# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import os
import pytest
import importlib
import learnrl

@pytest.fixture
def reload_tensorflow():
    yield
    importlib.reload(learnrl.callbacks)
    importlib.reload(learnrl)

@pytest.fixture
def hide_tensorflow(reload_tensorflow, mocker):
    mocker.patch.dict('sys.modules', { 'tensorflow': None })
    importlib.reload(learnrl.callbacks)
    importlib.reload(learnrl)

@pytest.fixture
def reload_wandb():
    yield
    importlib.reload(learnrl.callbacks)
    importlib.reload(learnrl)

@pytest.fixture
def hide_wandb(reload_wandb, mocker):
    mocker.patch.dict('sys.modules', { 'wandb': None })
    importlib.reload(learnrl.callbacks)
    importlib.reload(learnrl)
