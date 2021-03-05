# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from typing import Dict, Union
import numpy as np

class Agent():

    """A general structure for any learning agent."""

    def act(self, observation, greedy: bool=False) -> Dict[Union[int, float, np.ndarray]]:
        """How the :ref:`Agent` act given an observation.

        Args:
            observation: The observation given by the |gym.Env|.
            greedy: If True, act greedely (without exploration).

        """
        raise NotImplementedError

    def learn(self) -> Dict[Union[int, float, np.ndarray]]:
        """How the :ref:`Agent` learns from his experiences.

        Returns:
            logs: The agent learning logs (Has to be numpy or python).

        """
        return {}

    def remember(self,
            observation,
            action,
            reward,
            done,
            next_observation=None,
            info=None,
            **param
        ):
        """How the :ref:`Agent` will remember experiences.

        Often, the agent will use a |hash| to store observations efficiently.

        Example:
            >>>  self.memory.remember(self.observation_encoder(observation),
            ...                       self.action_encoder(action),
            ...                       reward, done,
            ...                       self.observation_encoder(next_observation),
            ...                       info, **param)

        """
        pass
