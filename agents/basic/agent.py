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

    action_values = {}
    action_visits = {}

    def __init__(self, evaluation=MonteCarlo(initial_learning_rate=0.1), control=Greedy(initial_exploration=1.0, decay=0.999)):
        super().__init__()
        self.evaluation = evaluation
        self.control = control
        self.name = self.name + '_{}'.format(self.evaluation.name) + '_{}'.format(self.control.name)
        self.memory.legal_actions = {}

    @staticmethod
    def _hash_state(state):
        try:
            state_id = hash(state)
        except TypeError:
            state_id = hash(state.tostring())
        return state_id

    @staticmethod
    def _hash_action(action):
        try:
            action_id = hash(action)
        except TypeError:
            action_id = hash(action.tostring())
        return action_id

    def policy(self, state, legal_actions):     
        try:
            N = np.array([self.action_visits[(state, action)] for action in legal_actions])
            Q = np.array([self.action_values[(state, action)] for action in legal_actions])
            policy = self.control.getPolicy(action_visits=N, action_values=Q)
        except KeyError:
            policy = np.ones(legal_actions.shape)/legal_actions.shape[-1]
        return policy
    
    def act(self, state, legal_actions):
        state_id = self._hash_state(state)
        legal_actions_id = np.array([self._hash_action(action) for action in legal_actions])
        self.memory.legal_actions[state_id] = legal_actions_id

        policy = self.policy(state_id, legal_actions_id)
        action_id = np.random.choice(range(len(legal_actions)), p=policy) # pylint: disable=E1136  # pylint/issues/3139
        return action_id
    
    def remember(self, state, action, reward, done, next_state=None, info={}):
        self.memory.remember(self._hash_state(state), self._hash_action(action), reward, done, self._hash_state(next_state), info)

    def learn(self, **kwargs):
        self.evaluation.learn(action_values=self.action_values, action_visits=self.action_visits, memory=self.memory, policy=self.policy, **kwargs)
        self.control.updateExploration()
        self.evaluation.update_learning_rate()
