# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from learnrl.callbacks.callback import Callback, CallbackList

from learnrl.callbacks.logging_callback import LoggingCallback, MetricList, Metric
from learnrl.callbacks.logger import Logger

import importlib
tensorflow_spec = importlib.util.find_spec('tensorflow')

if tensorflow_spec is not None:
    from learnrl.callbacks.tensorboard import TensorboardCallback
else:
    class TensorboardCallback():
        def __init__(self, log_dir):
            raise ImportError('Missing dependency : tensorflow >= 2.0.0')

wandb_spec = importlib.util.find_spec('wandb')

if wandb_spec is not None:
    from learnrl.callbacks.wandb import WandbLogger
else:
    class WandbLogger():
        def __init__(self):
            raise ImportError('Missing dependency : wandb >= 0.10')

