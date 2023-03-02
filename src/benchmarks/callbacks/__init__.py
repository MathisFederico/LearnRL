# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""Callbacks to perform actions while running a playground"""

import importlib.util

from benchmarks.callbacks.callback import Callback, CallbackList
from benchmarks.callbacks.logging_callback import LoggingCallback, MetricList, Metric
from benchmarks.callbacks.logger import Logger

tensorflow_spec = importlib.util.find_spec('tensorflow')
if tensorflow_spec is not None:
    from benchmarks.callbacks.tensorboard import TensorboardCallback
else:
    class TensorboardCallback():
        """Dummy TensorboardCallback when tensorflow in not installed"""
        def __init__(self, log_dir):
            raise ImportError('Missing dependency : tensorflow >= 2.0.0')

wandb_spec = importlib.util.find_spec('wandb')
if wandb_spec is not None:
    from benchmarks.callbacks.wandb import WandbCallback
else:
    class WandbCallback():
        """Dummy WandbCallback when wandb in not installed"""

        def __init__(self, run=None):
            raise ImportError('Missing dependency : wandb >= 0.10')
