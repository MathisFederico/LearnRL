# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
import learnrl
import importlib
tensorflow_spec = importlib.util.find_spec('tensorflow')

if tensorflow_spec is not None:

    class TestTensorboardCallback():

        def test_instanciate_without_tensorflow(self, hide_tensorflow): # pylint: disable=unused-argument
            """Should raise ImportError if tensorflow is not found"""
            with pytest.raises(ImportError, match=r".*tensorflow >= 2.*"):
                learnrl.callbacks.TensorboardCallback(log_dir='logs')

        def test_instanciate_with_tensorflow(self):
            """Should instanciate correctly if tensorflow is found"""
            learnrl.callbacks.TensorboardCallback(log_dir='logs')
else:
    pass
