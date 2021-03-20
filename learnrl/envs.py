# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""Environment types, extentions from the classic gym.Env"""

from abc import abstractmethod
from gym import Env

class TurnEnv(Env):

    r"""Turn based multi-agents gym environment.

    A layer over the gym |gym.Env| class able to handle
    turn based environments with multiple agents.

    .. note::
        A :ref:`TurnEnv` must be in a :ref:`Playground` in order to work !

    The only add in TurnEnv is the method "turn",
    On top of the main API basic methodes (see |gym.Env|):
        * step: take a step of the |gym.Env| given the action of the active player
        * reset: reset the |gym.Env| and returns the first observation
        * render
        * close
        * seed

    Attributes:
        action_space (|gym.Space|): The Space object corresponding to actions.
        observation_space (|gym.Space|): The Space object corresponding to observations.

    """

    @abstractmethod
    def step(self, action):
        """Perform a step of the environement.

        Args:
            action: The action taken by the agent who's turn was given by :meth:`turn`.

        Returns:
            observation: The observation to give to the :class:`~learnrl.agent.Agent`.
            reward (float): The reward given to the :class:`~learnrl.agent.Agent` for this step.
            done (bool): True if the environement is done after this step.
            info (dict): Additional informations given by the |gym.Env|.

        """
        raise NotImplementedError('TurnEnv.step must be user-defined by subclassing.')

    @abstractmethod
    def turn(self, state) -> int:
        """Give the turn to the next agent to play.

        Assuming that agents are represented by a list like range(n_player)
        where n_player is the number of players in the game.

        Args:
            state: The state of the environement.
                Should be enough to determine which is the next agent to play.

        Returns:
            agent_id (int): The next player id

        """
        raise NotImplementedError('TurnEnv.turn must be user-defined by subclassing.')

    @abstractmethod
    def reset(self):
        """Reset the environement and returns the initial state.

        Returns:
            observation: The observation for the first :class:`Agent` to play

        """
        raise NotImplementedError('TurnEnv.reset must be user-defined by subclassing.')
