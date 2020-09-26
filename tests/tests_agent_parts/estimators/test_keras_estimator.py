# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
from gym.spaces import Discrete

import importlib
tensorflow_spec = importlib.util.find_spec('tensorflow')

if tensorflow_spec is not None:

    def test_instanciate_keras_estimator_without_tensorflow(hide_tensorflow):
        from learnrl.estimators import KerasEstimator

        class DummyEstimator(KerasEstimator):
            def build(self, **kwargs):
                pass
        
        observation_space = Discrete(4)
        action_space = Discrete(3)
        with pytest.raises(ImportError, match=r".*tensorflow >= 2.*"):
            DummyEstimator(observation_space, action_space)

    def test_instanciate_keras_estimator_with_tensorflow():
        from learnrl.estimators import KerasEstimator

        class DummyEstimator(KerasEstimator):
            def build(self, **kwargs):
                pass
        
        observation_space = Discrete(4)
        action_space = Discrete(3)
        DummyEstimator(observation_space, action_space)
