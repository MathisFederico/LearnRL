""" Configure tests and define markers """

# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import importlib
import pytest
import learnrl

def pytest_addoption(parser):
    """ Add options to pytest CLI """
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    """ Register new markers """
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    """ Skip tests according to slow parameter """

    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

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
