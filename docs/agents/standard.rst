StandardAgent
=============

StandardAgent are composed of three :ref:`Agent_parts`:
- :ref:`Control`
- :ref:`Evaluation`
- :ref:`Estimator`

They are all linked by the :class:`~learnrl.agents.StandardAgent`, as showned by this:

.. raw:: html
   :file: ../_static/images/StandardAgent.svg


TableAgent
----------

| Table agents are the simplest form of RL Agents.
| With experience, we build an action_value array (Q in literature).
| Q(s,a) being the expected futur rewards given that the agent did the action a in the state s.
| For that, they are composed of two main objects : Control and Evaluation

| A :class:`~learnrl.agent_parts.control.Control` object uses the action_value to determine the policy or behavior of the agent.
| An :class:`~learnrl.agent_parts.evaluation.Evaluation` object predicts the expected rewards from an experience given a behavior.

| By default, when building a :class:`~learnrl.agents.StandardAgent`, it will be a TableAgent (using a :class:`~learnrl.agent_parts.esimator.TableEstimator`) 
| with eps-greedy control (see :ref:`Control`)
| and QLearning evaluation (see :ref:`Evaluation`)

Here is an example :

.. code-block:: python

   from learnrl.agents import StandardAgent

   agent_table = StandardAgent(observation_space=observation_space,
                               action_space=action_space,
                               exploration=0.1,
                               exploration_decay=1e-2,
                               learning_rate=0.1)


DeepRLAgent
-----------

| If we replace the :class:`~learnrl.agent_parts.esimator.TableEstimator` by any other estimator, we can build approximations of Q(s,a).
| This is way more suitable in general because the :class:`~learnrl.agent_parts.esimator.TableEstimator` needs too much ram for real applications.
| For example, replacing the action_value by a :class:`~learnrl.agent_parts.esimator.KerasEstimator` and completing the methods build and preprocess like so :

.. code-block:: python

   from learnrl.agent_parts.estimator import KerasEstimator
   from learnrl.agents import StandardAgent

   class MyEstimator(KerasEstimator):

    def build(self):
        
        self.model = Sequential()
        self.model.add(Flatten(input_shape=self.observation_shape))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(self.action_size))

        self.model.compile(Adam(learning_rate=self.learning_rate), loss='mse')

    def preprocess(self, observations, actions=None):
        X = copy(observations)
        X[X==2] = -1
        return X
   
   custom_action_value = MyEstimator(observation_space=observation_space,
                                     action_space=action_space,
                                     epochs_per_step=10,
                                     batch_size=32,
                                     learning_rate=1e-2,
                                     freezed_steps=100,
                                     verbose=0)
   
   agent_deep = StandardAgent(observation_space=observation_space,
                              action_space=action_space,
                              action_values=custom_action_value)


| You built a neural network to make an approximation of Q(s,a) !

.. autoclass:: learnrl.agents.StandardAgent
   :members:
   :undoc-members:

.. |gym.Env| replace:: `environment <http://gym.openai.com/docs/#environments>`__
.. |gym.Space| replace:: `space <http://gym.openai.com/docs/#spaces>`__
.. |hash| replace:: `perfect hash functions <https://en.wikipedia.org/wiki/Perfect_hash_function>`__
