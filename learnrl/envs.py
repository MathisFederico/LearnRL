from gym import Env

class TurnEnv(Env):

    r"""
    A layer over the gym |gym.Env| class able to handle turn based environements with multiple agents.
   
    .. note::
        A :ref:`TurnEnv` must be in a :ref:`Playground` in order to work !

    | The main add in TurnEnv is the method: 
    |   turn

    On top of the main API basic methodes (see |gym.Env|):
        * step: take a step of the |gym.Env| given the action of the active player
        * reset: reset the |gym.Env| and returns the first observation
        * render
        * close 
        * seed

    Attributes
    ----------
        action_space: |gym.Space|
            The Space object corresponding to actions  
        observation_space: |gym.Space|
            The Space object corresponding to observations  
        reward_range: :class:`tuple`
            | A tuple corresponding to the min and max possible rewards.
            | A default reward range set to [-inf,+inf] already exists. 
            | Set it if you want a narrower range.

    """

    def step(self, action):
        """Perform a step of the environement
        
        Parameters
        ----------
            action:
                The action taken by the agent who's turn was given by :meth:`turn`.
        
        Return
        ------
            observation: 
                The observation to give to the :class:`Agent`.
            reward: :class:`float`
                The reward given to the :class:`Agent` for this step.
            done: :class:`bool`
                Is the environement done after this step ?
            info: :class:`dict`
                Additional informations given by the |gym.Env|.
            
        """
        raise NotImplementedError

    def turn(self, state):
        """Give the turn to the next agent to play
    
        Assuming that agents are represented by a list like range(n_player)
        where n_player is the number of players in the game.
        
        Parameters
        ----------
            state: 
                | The real state of the environement.
                | Should be enough to determine which is the next agent to play.

        Return
        ------
            agent_id: :class:`int`
                The next player id

        """
        raise NotImplementedError

    def reset(self):
        """Reset the environement and returns the initial state

        Return
        ------
            observation:
                The observation for the first :class:`Agent` to play
        
        """
        raise NotImplementedError


