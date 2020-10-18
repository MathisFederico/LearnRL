<<<<<<< HEAD
from learnrl.memory import Memory
=======

>>>>>>> e8418a7ba8405219567800c97244293b9e6df5a5

class Agent():

    """ A general structure for reinforcement learning agents    
    
<<<<<<< HEAD
    It uses by default a :class:`~learnl.memory.Memory`
=======
    It uses by default a :class:`Memory`
>>>>>>> e8418a7ba8405219567800c97244293b9e6df5a5

    Attributes
    ----------

        name: :class:`str`
            The Agent's name
<<<<<<< HEAD
        memory: :class:`~learnl.memory.Memory`
=======
        memory: :class:`Memory`
>>>>>>> e8418a7ba8405219567800c97244293b9e6df5a5
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
<<<<<<< HEAD
        """ Uses the agent's :class:`~learnl.memory.Memory` to remember experiences
=======
        """ Uses the agent's :class:`Memory` to remember experiences
>>>>>>> e8418a7ba8405219567800c97244293b9e6df5a5
        
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
<<<<<<< HEAD
    
=======
    
>>>>>>> e8418a7ba8405219567800c97244293b9e6df5a5
