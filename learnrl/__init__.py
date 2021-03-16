# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""LearnRL a python library to learn and log reinforcement learning"""

__version__ = "1.0.0"

from learnrl.agent import Agent
from learnrl.envs import TurnEnv
from learnrl.playground import Playground, DoneHandler, RewardHandler
from learnrl.callbacks import Callback
