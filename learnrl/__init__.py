# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

__version__ = "0.2.0"

from learnrl.agent import Agent
from learnrl.envs import TurnEnv
from learnrl.playground import Playground, DoneHandler, RewardHandler
from learnrl.memory import Memory

from learnrl.estimators import Estimator
from learnrl.control import Control
from learnrl.evaluation import Evaluation
