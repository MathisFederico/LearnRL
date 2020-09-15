# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
import importlib
import learnrl

def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

@pytest.fixture
def reload_tensorflow():
    yield
    importlib.reload(learnrl.callbacks)
    importlib.reload(learnrl.estimators)
    importlib.reload(learnrl)

@pytest.fixture
def hide_tensorflow(reload_tensorflow, mocker):
    mocker.patch.dict('sys.modules', { 'tensorflow': None })
    importlib.reload(learnrl.callbacks)
    importlib.reload(learnrl.estimators)
    importlib.reload(learnrl)

