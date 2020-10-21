class Agent():

    """ A general structure for reinforcement learning agents    
    
    It uses by default a :class:`~learnl.memory.Memory`

    Attributes
    ----------

        name: :class:`str`
        memory: :class:`~learnl.memory.Memory`
            The Agent's memory
    
    """

    def act(self, observation, greedy=False):
        """ How the :ref:`Agent` act given an observation
        
        Parameters
        ----------
            observation:
                The observation given by the |gym.Env|
            greedy: bool
                If True, act greedely (without exploration)

        """
        raise NotImplementedError

    def learn(self):
        """ How the :ref:`Agent` learns from his experiences 
        
        Returns
        -------
            logs: dict
                The agent learning logs.

        """
        raise NotImplementedError

    def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
        """ Uses the agent's :class:`~learnl.memory.Memory` to remember experiences
        
        Often, the agent will use a |hash| to store observations efficiently

        Example
        -------
            >>>  self.memory.remember(self.observation_encoder(observation),
            ...                       self.action_encoder(action),
            ...                       reward, done, 
            ...                       self.observation_encoder(next_observation), 
            ...                       info, **param)
        """
        raise NotImplementedError
