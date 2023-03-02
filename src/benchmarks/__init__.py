# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""LearnRL a python library to learn and log reinforcement learning"""

__version__ = "1.0.2"

from benchmarks.agent import Agent
from benchmarks.envs import TurnEnv
from benchmarks.playground import Playground, DoneHandler, RewardHandler
from benchmarks.callbacks import Callback
