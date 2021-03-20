Welcome to LearnRL's community !
================================

.. image:: https://img.shields.io/pypi/l/learnrl
   :alt: PyPI - License
   :target: https://www.gnu.org/licenses/


.. image:: https://app.codacy.com/project/badge/Grade/b4c3818135484e8b9acae67b01526957
   :alt: Codacy - Quality
   :target: https://www.codacy.com/gh/MathisFederico/LearnRL/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=MathisFederico/LearnRL&amp;utm_campaign=Badge_Grade


.. image:: https://app.codacy.com/project/badge/Coverage/b4c3818135484e8b9acae67b01526957
   :alt: Codacy - Coverage
   :target: https://www.codacy.com/gh/MathisFederico/LearnRL/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=MathisFederico/LearnRL&amp;utm_campaign=Badge_Coverage


LearnRL is a library to use and learn reinforcement learning.
It's also a community off supportive enthousiasts loving to share and build RL-based AI projects !
We would love to help you make projects with LearnRL, so join us `on Discord <https://discord.gg/z9dd4s5>`_ !

About LearnRL
-------------

LearnRL is a tool to monitor and log reinforcement learning experiments.
You build/find any compatible agent (only need an act method), you build/find a gym environment, and learnrl will make them interact together !
LearnRL also contains both tensorboard and weights&biases integrations for a beautiful and sharable experiment tracking !  
Also, LearnRL is cross platform compatible ! That's why no agents are built-in learnrl itself, but you can check:
   - `LearnRL for Tensorflow <https://github.com/MathisFederico/LearnRL-Tensorflow>`_
   - `LearnRL for Pytorch <https://github.com/MathisFederico/LearnRL-Pytorch>`_

You can build and run your own Agent in a clear and sharable manner !

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
         ...

   env = gym.make('FrozenLake-v0', is_slippery=True) # This could be any gym Environment !
   agent = MyAgent(env.observation_space, env.action_space)

   pg = rl.Playground(env, agent)
   pg.fit(2000, verbose=1)

Note that 'learn' and 'remember' are optional, so this framework can also be used for baselines !

You can logs any custom metrics that your Agent/Env gives you and even chose how to aggregate them through different timescales.
See the `metric codes <https://learnrl.readthedocs.io/en/latest/callbacks.html#metric-codes>`_ for more details.

.. code-block:: python

   metrics=[
        ('reward~env-rwd', {'steps': 'sum', 'episode': 'sum'}),
        ('handled_reward~reward', {'steps': 'sum', 'episode': 'sum'}),
        'value_loss~vloss',
        'actor_loss~aloss',
        'exploration~exp'
    ]

   pg.fit(2000, verbose=1, metrics=metrics)

The Playground will allow you to have clean logs adapted to your will with the verbose parameter:
  - Verbose 1 : episodes cycles - If your environment makes a lot of quick episodes.
   .. image:: docs\_static\images\logs-verbose-1.png

  - Verbose 2 : episode - To log each individual episode.
   .. image:: docs\_static\images\logs-verbose-2.png

  - Verbose 3 : steps cycles - If your environment makes a lot of quick steps but has long episodes.
   .. image:: docs\_static\images\logs-verbose-3.png

  - Verbose 4 : step - To log each individual step.
   .. image:: docs\_static\images\logs-verbose-4.png

  - Verbose 5 : detailled step - To debug each individual step (with observations, actions, ...).
   .. image:: docs\_static\images\logs-verbose-5.png


The Playground also allows you to add Callbacks with ease, for example the WandbCallback to have a nice experiment tracking dashboard !


Installation
------------

Install LearnRL by running::

   pip install learnrl


Get started
-----------

Create:
   - TODO: Numpy tutorials
   - TODO: Tensorflow tutorials
   - TODO: Pytorch tutorials


Visualize:
   - TODO: Tensorboard visualisation tutorial
   - TODO: Wandb visualisation tutorial
   - TODO: Wandb sweeps tutorial


Documentation
-------------

.. image:: docs\_static\images\docs.png
   :target: https://learnrl.readthedocs.io/en/latest/


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
