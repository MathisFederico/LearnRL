# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
import learnrl
import importlib
tensorflow_spec = importlib.util.find_spec('tensorflow')

if tensorflow_spec is not None:
    def test_instanciate_without_tensorflow(hide_tensorflow):
        with pytest.raises(ImportError, match=r".*tensorflow >= 2.*"):
            callback = learnrl.callbacks.TensorboardCallback()

    def test_instanciate_with_tensorflow():
        callback = learnrl.callbacks.TensorboardCallback()
else:
    pass

