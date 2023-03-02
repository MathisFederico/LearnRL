"""Agent to represent any learning or static agent"""

from typing import Dict, Union
from abc import abstractmethod

import numpy as np


class Agent:

    """A general structure for any learning agent."""

    @abstractmethod
    def act(self, observation, greedy: bool = False) -> Union[int, float, np.ndarray]:
        """How the :ref:`Agent` act given an observation.

        Args:
            observation: The observation given by the |gym.Env|.
            greedy: If True, act greedely (without exploration).

        """
        raise NotImplementedError("Agent.act must be user-defined by subclassing.")

    def learn(self) -> Dict[str, Union[int, float, np.ndarray]]:
        """How the :ref:`Agent` learns from his experiences.

        Returns:
            logs: The agent learning logs (Has to be numpy or python).

        """
        return {}

    def remember(
        self,
        observation,
        action,
        reward,
        done,
        next_observation=None,
        info=None,
        **param,
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
