# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from learnrl.estimators.estimator import Estimator
from learnrl.estimators.numpy import TableEstimator

import importlib
tensorflow_spec = importlib.util.find_spec('tensorflow')

if tensorflow_spec is not None:
    from learnrl.estimators.tensorflow import KerasEstimator
else:
    class KerasEstimator():
        def __init__(self, observation_space, action_space, **kwars):
            raise ImportError('Missing dependency : tensorflow >= 2.0.0')
