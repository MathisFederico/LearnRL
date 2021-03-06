LearnRL's Core
==============

LearnRL is based on those core objects: :ref:`Playground`,
:ref:`Agent`, :ref:`TurnEnv`.

They are all linked by the :ref:`Playground`, as showned by this:

.. raw:: html
   :file: _static/images/Playground.svg


.. _Playground:

Playground
----------

.. autoclass:: learnrl.playground.Playground
   :members:


.. _Agent:

Agent
-----

.. autoclass:: learnrl.agent.Agent
   :members:


.. _TurnEnv:

TurnEnv
-------

.. autoclass:: learnrl.envs.TurnEnv
   :members:

Handlers
--------

RewardHandler
~~~~~~~~~~~~~

.. autoclass:: learnrl.playground.RewardHandler
   :members:


DoneHandler
~~~~~~~~~~~

.. autoclass:: learnrl.playground.DoneHandler
   :members:


.. |ndarray| replace:: :class:`numpy.ndarray`
.. |gym.Env| replace:: `environment <http://gym.openai.com/docs/#environments>`__
.. |gym.Space| replace:: `space <http://gym.openai.com/docs/#spaces>`__
.. |hash| replace:: `perfect hash functions <https://en.wikipedia.org/wiki/Perfect_hash_function>`__
