from agents.basic.evaluation import MonteCarlo, TemporalDifference
from agents.basic.control import UCB 
from agents.agent import Agent

import numpy as np
import collections

class BasicAgent(Agent):

    """ A general structure for table-based RL agents
    You can use different evaluation and control methods
    
    
    evaluations : agents.basic.evaluation
        'mc','montecarlo' -> Monte-Carlo evaluation
        'td','temp*dif*' -> Offline Temporal Difference
        'ontd','on*temp*dif' -> Online Temporal Difference
        'sarsa' -> SARSA with specified target policy
            You have to specify the target policy with #.target_policy = target_policy
        'q*' -> QLearning (SARSA with greedy target policy)
    

    control : agents.basic.control
        '*greedy' -> epsilon_greedy with epsilon=exploration
        'ucb' -> ucb with c=exploration
        'puct' -> puct with c=exploration
    """

    name = 'BasicAgent'

    action_values = {}
    action_visits = {}

    def __init__(self, evaluation=MonteCarlo(), control=UCB(), learning_rate=0.1, exploration_coef=1):
        self.evaluation = evaluation
        self.control = control
        self.name = self.name + '_{}'.format(self.evaluation.name) + '_{}'.format(self.control.name)
        self.exploration_coef = exploration_coef
        self.learning_rate = learning_rate

    def policy(self, observation):
        state_id = hash(observation)
        legal_action = np.array([]) # We need them from the environement

        try:
            N = np.array([self.action_visits[(state_id, action)] for action in legal_action])
            Q = np.array([self.action_values[(state_id, action)] for action in legal_action])
            policy = self.control.getPolicy(action_visits=N, action_values=Q, exploration_coef=self.exploration_coef)
        except KeyError:
            policy = np.ones(legal_action.shape)/legal_action.shape[-1] # pylint: disable=E1136  # pylint/issues/3139
        
        return policy
    
    def act(self, observation):
        policy = self.policy(observation)
        action_id = np.random.choice(range(policy.shape[-1]), policy) # pylint: disable=E1136  # pylint/issues/3139
        return action_id
    
    def learn(self):
        self.evaluation.learn(self.action_visits, self.action_values, self.memory, learning_rate=self.learning_rate)
