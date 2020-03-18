from agents.basic.evaluation import MonteCarlo, TemporalDifference
from agents.basic.control import Greedy 
from agents.agent import Agent

from warnings import warn

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

    name = 'basic'
    
    def __init__(self, state_size, action_size, control=None, evaluation=None, **kwargs):
        
        super().__init__()
        self.control = control if control is not None else Greedy(action_size, **kwargs)
        self.evaluation = evaluation if evaluation is not None else MonteCarlo(**kwargs)

        self.name = f'{self.name}_{self.control.name}_{self.evaluation.name}_{kwargs}'
        
        self.action_values = np.zeros((state_size, action_size))
        self.action_visits = np.zeros((state_size, action_size))
    
    def act(self, state, legal_actions, greedy=False):
        state_id = self._hash_state(state)
        policy = self.control.get_policy(state_id, self.action_values, self.action_visits)

        legal_actions_id = np.array([self._hash_action(action) for action in legal_actions])
        legal_policy = policy[legal_actions_id]

        action_id = np.random.choice(range(len(legal_actions)), p=legal_policy) # pylint: disable=E1136  # pylint/issues/3139
        return action_id
    
    def remember(self, state, action, reward, done, next_state=None, info={}):
        self.memory.remember(self._hash_state(state), self._hash_action(action), reward, done, self._hash_state(next_state), info)

    def learn(self, **kwargs):
        self.evaluation.learn(action_values=self.action_values, action_visits=self.action_visits, memory=self.memory, policy=self.control.get_policy, **kwargs)
        self.control.update_exploration()
        self.evaluation.update_learning_rate()

    def __str__(self):
        return self.name
    
    def __call__(self, state, legal_actions, greedy=False):
        return self.act(state, legal_actions, greedy=False)

    @staticmethod
    def _hash_state(state):
        if type(state) == np.ndarray:
            state_id = hash(state.tostring())
        else:
            state_id = hash(state)
        return state_id

    @staticmethod
    def _hash_action(action):
        if type(action) == np.ndarray:
            action_id = hash(action.tostring())
        else:
            action_id = hash(action)
        return action_id

class QLearningAgent(BasicAgent):

    name = 'qlearning'

    def __init__(self, state_space, action_space, control=None, evaluation=None, **kwargs):

        evaluation = TemporalDifference
        target_control = kwargs.get('target_control', None)
        if target_control is not None:
            raise warn(r"'target_control' kwarg shouldn't be specified for QLearningAgent")
        target_control = Greedy
        
        super().__init__(state_space, action_space, control=None, evaluation=None, target_control=target_control, **kwargs)
