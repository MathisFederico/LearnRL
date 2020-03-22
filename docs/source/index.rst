Welcome to LearnRL's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   learnrl
   learnrl.agents
   learnrl.environments

| LearnRL is a librairie to use and learn reinforcement learning.
| Look how easy it is to use:

.. code-block:: python
   :linenos:

   import learnrl as rl
   from learnrl.environments import CrossesAndNoughtsEnv
   from learnrl.agents import TableAgent

   env = CrossesAndNoughtsEnv()
   agent1 = TableAgent(state_space=env.observation_space, action_space=env.action_space)
   agent2 = TableAgent(state_space=env.observation_space, action_space=env.action_space)

   agents = [agent1, agent2]
   pg = rl.Playground(env, agents)
   pg.fit(50000, verbose=1)

And boom you made two QLearning AIs training against each other on crosses and noughts !

Features
--------

- Build highly configurable classic reinforcement learning agents in few lines of code
- Train your Agents on any Gym environments
- Use this API to create your own agents and environments (even multiplayer!) with great compatibility

Installation
------------

Install LearnRL by running:

   pip install learnrl

Contribute
----------

- Issue Tracker : github.com/MathisFederico/LearnRL/issues
- Source Code : github.com/MathisFederico/LearnRL

Support
-------

If you are having issues, please let me know at mathfederico@gmail.com

License
-------

| The project is licensed under the GNU LGPLv3 license.
| See LICENCE, COPYING and COPYING.LESSER for more details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
