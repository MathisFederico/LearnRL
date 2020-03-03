from agents.basic.evaluation import MonteCarlo, TemporalDifference
from agents.basic.control import Greedy 
from agents.agent import Agent

import numpy as np

class BasicAgent(Agent):

    """ 
    A general structure for table-based RL agents.
    You can use different evaluation and control methods.
    
    Evaluations : agents.basic.evaluation
        'mc','montecarlo' -> Monte-Carlo evaluation
        X'sarsa' -> SARSA with specified target policy
        X'q*' -> QLearning (SARSA with greedy target policy)
        
    Control : agents.basic.control
        'greedy' -> epsilon_greedy with epsilon=exploration
        'ucb' -> ucb with c=exploration
        'puct' -> puct with c=exploration
    """

    name = 'BasicAgent'
    def __init__(self, state_size, action_size,
                       evaluation=MonteCarlo(initial_learning_rate=0.1), 
                       control=Greedy(initial_exploration=1.0, decay=0.999)):
        super().__init__()
        self.evaluation = evaluation
        self.control = control
        self.name = self.name + '_{}'.format(self.evaluation.name) + '_{}'.format(self.control.name)
        self.action_values = np.zeros((state_size, action_size))
        self.action_visits = np.zeros((state_size, action_size))
        self._hash_state = np.vectorize(self._hash_state)
        self._hash_action = np.vectorize(self._hash_action)

    @staticmethod
    def _hash_state(state): # pylint: disable=E0202
        if type(state) == np.ndarray:
            state_id = hash(state.tostring())
        else:
            state_id = hash(state)
        return state_id

    @staticmethod
    def _hash_action(action): # pylint: disable=E0202
        if type(action) == np.ndarray:
            action_id = hash(action.tostring())
        else:
            action_id = hash(action)
        return action_id

    def policy(self, state):
        N = self.action_visits[state, :]
        Q = self.action_values[state, :]
        return self.control.getPolicy(action_visits=N, action_values=Q)
    
    def act(self, state, legal_actions):
        state_id = self._hash_state(state)
        policy = self.policy(state_id)

        legal_actions_id = np.array([self._hash_action(action) for action in legal_actions])
        legal_policy = policy[legal_actions_id]

        action_id = np.random.choice(range(len(legal_actions)), p=legal_policy) # pylint: disable=E1136  # pylint/issues/3139
        return action_id
    
    def remember(self, state, action, reward, done, next_state=None, info={}):
        self.memory.remember(self._hash_state(state), self._hash_action(action), reward, done, self._hash_state(next_state), info)

    def learn(self, **kwargs):
        self.evaluation.learn(action_values=self.action_values, action_visits=self.action_visits, memory=self.memory, policy=self.policy, **kwargs)
        self.control.updateExploration()
        self.evaluation.update_learning_rate()
