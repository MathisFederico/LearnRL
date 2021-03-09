# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" TensorboardCallback """

import importlib.util
import pytest

import learnrl

tensorflow_spec = importlib.util.find_spec('tensorflow')

if tensorflow_spec is not None:
    class TestTensorboardCallback():
        """ TensorboardCallback """

        def test_instanciate_without_tensorflow(self, hide_tensorflow):
            """ should raise an ImportError when tensorflow is not found. """
            with pytest.raises(ImportError, match=r".*tensorflow >= 2.*"):
                learnrl.callbacks.TensorboardCallback(log_dir='logs')

        def test_instanciate_with_tensorflow(self):
            """ should instanciate correctly when tensorflow is found. """
            learnrl.callbacks.TensorboardCallback(log_dir='logs')
else:
    pass
