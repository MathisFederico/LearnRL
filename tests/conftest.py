""" Configure tests and define markers """


# pylint: disable=unused-argument

import importlib
import pytest
import benchmarks


@pytest.fixture
def reload_tensorflow():
    """Reload tensorflow after hiding it from the modules list"""

    yield
    importlib.reload(benchmarks.callbacks)
    importlib.reload(benchmarks)


@pytest.fixture
def hide_tensorflow(reload_tensorflow, mocker):
    """Hide tensorflow from the modules list"""
    mocker.patch.dict("sys.modules", {"tensorflow": None})
    importlib.reload(benchmarks.callbacks)
    importlib.reload(benchmarks)


@pytest.fixture
def reload_wandb():
    """Reload wandb after hiding it form the modules list"""
    yield
    importlib.reload(benchmarks.callbacks)
    importlib.reload(benchmarks)


@pytest.fixture
def hide_wandb(reload_wandb, mocker):
    """Hide wandb from the modules list"""
    mocker.patch.dict("sys.modules", {"wandb": None})
    importlib.reload(benchmarks.callbacks)
    importlib.reload(benchmarks)
