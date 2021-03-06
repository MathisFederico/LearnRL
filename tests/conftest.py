# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
import importlib
import learnrl

@pytest.fixture
def reload_tensorflow():
    """Reload tensorflow after it has been hidden"""
    yield
    importlib.reload(learnrl.callbacks)
    importlib.reload(learnrl)

@pytest.fixture
def hide_tensorflow(reload_tensorflow, mocker): # pylint: disable=unused-argument
    """Hide tensorflow package"""
    mocker.patch.dict('sys.modules', { 'tensorflow': None })
    importlib.reload(learnrl.callbacks)
    importlib.reload(learnrl)

@pytest.fixture
def reload_wandb():
    """Reload wandb after it has been hidden"""
    yield
    importlib.reload(learnrl.callbacks)
    importlib.reload(learnrl)

@pytest.fixture
def hide_wandb(reload_wandb, mocker): # pylint: disable=unused-argument
    """Hide wandb package"""
    mocker.patch.dict('sys.modules', { 'wandb': None })
    importlib.reload(learnrl.callbacks)
    importlib.reload(learnrl)
