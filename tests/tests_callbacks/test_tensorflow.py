# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
import sys

def test_import(mocker):
    mocker.patch.dict('sys.modules', { 'tensorflow': None })
    with pytest.raises(ImportError, match=r".*tensorflow >= 2.*"):
        from learnrl.callbacks.tensorflow import TensorboardCallback
