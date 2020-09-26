TableAgent
==========

| Table agents are the simplest form of RL Agents.
| With experience, we build an action_value array (Q in literature).
| Q(s,a) being the expected futur rewards given that the agent did the action a ins the state s.
| For that, they are composed of two main objects : Control and Evaluation

| A :ref:`Control` object uses the action_value to determine the probabilities of choosing every action
| An :ref:`Evaluation` object uses the experience of the :any:`TableAgent <learnrl.agents.table>`
 present in his :ref:`Memory`.

.. autoclass:: learnrl.agents.table.agent.TableAgent
   :members:
   :undoc-members:

.. _Control:

Control
-----------------------------------

.. automodule:: learnrl.agents.table.control
   :members:
   :undoc-members:

.. _Evaluation:

Evaluation
--------------------------------------

.. automodule:: learnrl.agents.table.evaluation
   :members:
   :undoc-members:


.. |gym.Env| replace:: `environment <http://gym.openai.com/docs/#environments>`__
.. |gym.Space| replace:: `space <http://gym.openai.com/docs/#spaces>`__
.. |hash| replace:: `perfect hash functions <https://en.wikipedia.org/wiki/Perfect_hash_function>`__
