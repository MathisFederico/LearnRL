# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

def get_version():
    version_file = open('../VERSION')
    return version_file.read().strip()
__version__ = get_version()

from learnrl.core import *
import learnrl.environments
import learnrl.agents
