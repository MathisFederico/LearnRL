LearnRL's Core
==============

LearnRL is base on those core objects: :ref:`Playground`,
:ref:`Agent`, :ref:`MultiEnv`, :ref:`Memory`.

They are all linked by the :ref:`Playground`, as showned by this:

.. raw:: html
   :file: _static/images/LearnRL-Playground.svg

| Legend : (O:observation, A:action, R:reward, D:done, O':next_observartion, I:info)

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

.. |ndarray| replace:: :class:`numpy.ndarray`

Memory
------

.. autoclass:: learnrl.core.Memory
   :members:

.. |gym.Env| replace:: `environment <http://gym.openai.com/docs/#environments>`__
.. |gym.Space| replace:: `space <http://gym.openai.com/docs/#spaces>`__
.. |hash| replace:: `perfect hash functions <https://en.wikipedia.org/wiki/Perfect_hash_function>`__
