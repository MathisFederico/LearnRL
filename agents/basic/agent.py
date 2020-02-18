from agents.basic.evaluation import MonteCarlo, TemporalDifference
from agents.basic.control import Greedy 
from agents.agent import Agent

import numpy as np
import collections

class BasicAgent(Agent):

    """ 
    A general structure for table-based RL agents.
    You can use different evaluation and control methods.
    
    Evaluations : agents.basic.evaluation
        'mc','montecarlo' -> Monte-Carlo evaluation
        X'td','temp*dif*' -> Offline Temporal Difference
        X'ontd','on*temp*dif' -> Online Temporal Difference
        X'sarsa' -> SARSA with specified target policy
        X'q*' -> QLearning (SARSA with greedy target policy)
        
    Control : agents.basic.control
        '*greedy' -> epsilon_greedy with epsilon=exploration
        'ucb' -> ucb with c=exploration
        'puct' -> puct with c=exploration
    """

    name = 'BasicAgent'

    action_values = {}
    action_visits = {}

    def __init__(self, evaluation=MonteCarlo(), control=Greedy(initial_exploration=1.0, decay=0.999), learning_rate=0.1, exploration=0):
        self.evaluation = evaluation
        self.control = control
        self.name = self.name + '_{}'.format(self.evaluation.name) + '_{}'.format(self.control.name)
        self.exploration = exploration
        self.learning_rate = learning_rate

    def policy(self, observation, legal_actions):
        state_id = hash(observation)
        try:
            N = np.array([self.action_visits[(state_id, action)] for action in legal_actions])
            Q = np.array([self.action_values[(state_id, action)] for action in legal_actions])
            policy = self.control.getPolicy(action_visits=N, action_values=Q, exploration=self.exploration)
        except KeyError:
            policy = np.ones(legal_actions.shape)/legal_actions.shape[-1]
        
        return policy
    
    def act(self, observation, legal_actions):
        policy = self.policy(observation, legal_actions)
        action_id = np.random.choice(range(len(legal_actions)), p=policy) # pylint: disable=E1136  # pylint/issues/3139
        return action_id
    
    def learn(self):
        self.control.updateExploration()
        self.evaluation.learn(self.action_visits, self.action_values, self.memory, learning_rate=self.learning_rate)
