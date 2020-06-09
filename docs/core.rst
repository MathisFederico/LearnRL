LearnRL's Core
==============

LearnRL is based on those core objects: :ref:`Playground`,
:ref:`Agent`, :ref:`MultiEnv`, :ref:`Memory`.

They are all linked by the :ref:`Playground`, as showned by this:

.. raw:: html
   :file: _static/images/Playground.svg

.. _Playground:

Playground
----------

.. autoclass:: learnrl.core.Playground
   :members:

.. _Agent:

Agent
-----

.. autoclass:: learnrl.core.Agent
   :members:

.. _MultiEnv:

MultiEnv
--------

.. autoclass:: learnrl.core.MultiEnv
   :members:

.. _Memory:

Memory
------

.. autoclass:: learnrl.core.Memory
   :members:

.. |ndarray| replace:: :class:`numpy.ndarray`
.. |gym.Env| replace:: `environment <http://gym.openai.com/docs/#environments>`__
.. |gym.Space| replace:: `space <http://gym.openai.com/docs/#spaces>`__
.. |hash| replace:: `perfect hash functions <https://en.wikipedia.org/wiki/Perfect_hash_function>`__