TableAgent
============================

| Table agents are the simplest form of RL Agents.
| With experience, we build an action_value array (Q in literature).
| Q(s,a) being the expected futur rewards given that the agent did the action a ins the state s.
| For that, they are composed of two main objects : Control and Evaluation

| A :any:`Control` object uses the action_value to determine the probabilities of choosing every action
| An :any:`Evaluation` object uses the experience of the :any:`TableAgent`
 present in his :any:`Memory`.

.. autoclass:: learnrl.agents.table.agent.TableAgent
   :members:
   :undoc-members:

Control
-----------------------------------

.. automodule:: learnrl.agents.table.control
   :members:
   :undoc-members:
   :show-inheritance:

Evaluation
--------------------------------------

.. automodule:: learnrl.agents.table.evaluation
   :members:
   :undoc-members:
   :show-inheritance:
