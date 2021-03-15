""" Configure tests and define markers """

# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=unused-argument

import importlib
import pytest
import learnrl

@pytest.fixture
def reload_tensorflow():
    """ Reload tensorflow after hiding it from the modules list """

    yield
    importlib.reload(learnrl.callbacks)
    importlib.reload(learnrl)

@pytest.fixture
def hide_tensorflow(reload_tensorflow, mocker):
    """ Hide tensorflow from the modules list """
    mocker.patch.dict('sys.modules', { 'tensorflow': None })
    importlib.reload(learnrl.callbacks)
    importlib.reload(learnrl)

@pytest.fixture
def reload_wandb():
    """ Reload wandb after hiding it form the modules list """
    yield
    importlib.reload(learnrl.callbacks)
    importlib.reload(learnrl)

@pytest.fixture
def hide_wandb(reload_wandb, mocker):
    """ Hide wandb from the modules list """
    mocker.patch.dict('sys.modules', { 'wandb': None })
    importlib.reload(learnrl.callbacks)
    importlib.reload(learnrl)
