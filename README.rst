Welcome to LearnRL's documentation!
===================================

We would love to help you make projects with LearnRL, so join us `on Discord <https://discord.gg/z9dd4s5>`_ !

About LearnRL
-------------

LearnRL is a library to use and learn reinforcement learning.

It is very easy to use:

.. code-block:: python

   import learnrl as rl
   from learnrl.agents import StandardAgent

   import gym

   env = gym.make('FrozenLake-v0', is_slippery=True)
   agent = StandardAgent(env.observation_space, env.action_space,
                        exploration=1, exploration_decay=1e-4)

   pg = rl.Playground(env, agent)
   pg.fit(2000, verbose=1)

And very modular and customizable !
For example we can overide the evaluation method (how future rewards are expected)
and/or the control method (how a decision is made based on action value):

.. code-block:: python

   import learnrl as rl
   from learnrl.agents import StandardAgent

   import gym

   env = gym.make('FrozenLake-v0', is_slippery=True)

   class MyEvaluation(rl.Evaluation):

      """ MyEvaluation uses ... to approximate the expected return at each step. """

      def __init__(self, **kwargs):
         super().__init__(name="myevaluation", **kwargs)

      def eval(self, reward, done, next_observation, action_values, action_visits, control):
         ...
         return expected_futur_reward
      
   class MyControl(rl.Control):

      """ MyControl will make the policy given action values Q (and action visits N) """

      def __init__(self, exploration=1, **kwargs):
         super().__init__(exploration=exploration, name="mycontrol", **kwargs)
         self.need_action_visit = True # This is optional, here to ensure that N is given

      def policy(self, Q, N=None):
         ...
         return p

   evaluation = MyEvaluation(learning_rate=1e-2)
   control = MyControl(exploration=1, exploration_decay=1e-4)

   agent = StandardAgent(env.observation_space, env.action_space,
                        evaluation=evaluation, control=control)

   pg = rl.Playground(env, agent)
   pg.fit(2000, verbose=1)

You can of course build your own Agent and/or Environment from scratch !

.. code-block:: python

   import learnrl as rl
   import gym

   class MyAgent(rl.Agent):

      def act(self, observation, greedy=False):
         """ How the Agent act given an observation """
         ...
         return action

      def learn(self):
         """ How the Agent learns from his experiences """
         ...
         return logs

      def remember(self, observation, action, reward, done, next_observation=None, info={}, **param):
         """ How the Agent will remember experiences """
         pass

   env = gym.make('FrozenLake-v0', is_slippery=True)
   agent = MyAgent(env.observation_space, env.action_space)

   pg = rl.Playground(env, agent)
   pg.fit(2000, verbose=1)

Note that 'learn' and 'remember' are optional, so this can also be used for baselines.

Features
--------

- Build highly configurable classic reinforcement learning agents in few lines of code.
- Train your Agents on any Gym or custom environment.
- Use this API to create your own agents and environments (even multiplayer!) with great compatibility.

Installation
------------

Install LearnRL by running::

   pip install learnrl

Get started
----------

You can have a look at the cartpole example in the `examples` folder.

Documentation
-------------

| See the `latest complete documentation <https://learnrl.readthedocs.io/en/latest/>`_ for more details.
| See the `development documentation <https://learnrl.readthedocs.io/en/dev/>`_ to see what's coming !

Contribute
----------

- `Issue Tracker <https://github.com/MathisFederico/LearnRL/issues>`_.
- `Projects <https://github.com/MathisFederico/LearnRL/projects>`_.

Support
-------

If you are having issues, please contact us `on Discord <https://discord.gg/z9dd4s5>`_.

License
-------

| The project is licensed under the GNU LGPLv3 license.
| See LICENCE, COPYING and COPYING.LESSER for more details.

.. |gym.Env| replace:: `environment <http://gym.openai.com/docs/#environments>`__
.. |gym.Space| replace:: `space <http://gym.openai.com/docs/#spaces>`__
.. |hash| replace:: `perfect hash functions <https://en.wikipedia.org/wiki/Perfect_hash_function>`__
