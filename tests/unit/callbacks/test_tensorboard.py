""" Test Tensorboard callback """

# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import importlib
import pytest

import learnrl

tensorflow_spec = importlib.util.find_spec('tensorflow')

if tensorflow_spec is not None:
    def test_instanciate_without_tensorflow(hide_tensorflow):
        """ should raise an ImportError when tensorflow is not installed. """
        with pytest.raises(ImportError, match=r".*tensorflow >= 2.*"):
            learnrl.callbacks.TensorboardCallback(log_dir='logs')

    def test_instanciate_with_tensorflow():
        """ should load normally when tensorflow is installed. """
        learnrl.callbacks.TensorboardCallback(log_dir='logs')
else:
    pass
