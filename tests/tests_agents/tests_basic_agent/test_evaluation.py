import unittest
import numpy as np
from copy import deepcopy

class MonteCarloTest(unittest.TestCase):

    def test_nim_optimal_policy(self):
        from envs import NimEnv
        from agents import BasicAgent
        from agents.basic.evaluation import MonteCarlo

        env = NimEnv(is_optimal=True)
        agent = BasicAgent(evaluation=MonteCarlo())

        n_games = 2000
        legal_actions = np.array(range(3))

        for _ in range(n_games):
            done = False
            state = env.reset()
            while not done:
                action = agent.act(state, legal_actions)
                next_state, reward, done , info = env.step(action)
                agent.remember(state, action, reward, done, next_state, info)
                agent.learn()
                state = deepcopy(next_state)
        
        action_size, state_size = env.action_space.n, env.observation_space.n
        action_values = np.zeros((action_size,state_size))
        for action in range(action_size):
            for state in range(state_size):
                try:
                    action_values[action, state] = agent.action_values[(state, action)]
                except KeyError:
                    pass
        
        greedy_actions = 1+np.argmax(action_values, axis=0)
        pertinent_states = np.concatenate([[2+k+4*i for k in range(3)] for i in range(4)])
        pertinent_actions = greedy_actions[pertinent_states]
        expected_actions = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
        self.assertTrue(np.all(pertinent_actions==expected_actions))


class TemporalDifferenceTest(unittest.TestCase):

    def setUp(self):
        from agents import Memory
        self.memory = Memory()
        self.memory.remember(0, 0, 0, False, 1)
        self.memory.remember(1, 0, -1, True, 2)
        self.memory.legal_actions = {0:[0, 1], 1:[0, 1]}
        self.action_values = {(0,0):1, (0,1):0, (1, 0):1}
        self.action_visits = {(0,0):1, (0,1):1, (1, 0):1}

    def test_online_onpolicy(self):
        from agents.basic.evaluation import TemporalDifference
        from agents.basic.agent import BasicAgent

        agent = BasicAgent(evaluation=TemporalDifference())
        agent.memory = self.memory
        agent.action_values = self.action_values
        agent.action_visits = self.action_visits
        agent.learn(online=True)
        expected_action_visits = {(0, 0): 1, (0, 1): 1, (1, 0): 2}
        expected_action_values = {(0, 0): 1, (0, 1): 0, (1, 0): 0.8}
        expected_memory = {key:None for key in self.memory.MEMORY_KEYS}
        self.assertTrue(np.all([agent.action_visits[key]==expected_action_visits[key] for key in expected_action_visits]),
                        f"An error occured in action_visits\n \
                        Expected {expected_action_visits}\n \
                        Got {agent.action_visits}")
        self.assertTrue(np.all([agent.action_values[key]==expected_action_values[key] for key in expected_action_values]),
                        f"An error occured in action_values\n \
                        Expected {expected_action_values}\n \
                        Got {agent.action_values}")
        self.assertTrue(np.all([agent.memory.datas[key]==expected_memory[key] for key in expected_memory]),
                        f"An error occured in memory\n \
                        Expected {expected_memory}\n \
                        Got {agent.memory.datas}")
    
    def test_nim_optimal_policy(self):
        from envs import NimEnv
        from agents import BasicAgent
        from agents.basic.evaluation import TemporalDifference

        env = NimEnv(is_optimal=True)
        agent = BasicAgent(evaluation=TemporalDifference())

        n_games = 2000
        legal_actions = np.array(range(3))

        for _ in range(n_games):
            done = False
            state = env.reset()
            while not done:
                action = agent.act(state, legal_actions)
                next_state, reward, done , info = env.step(action)
                agent.remember(state, action, reward, done, next_state, info)
                agent.learn()
                state = deepcopy(next_state)
        
        action_size, state_size = env.action_space.n, env.observation_space.n
        action_values = np.zeros((action_size,state_size))
        for action in range(action_size):
            for state in range(state_size):
                try:
                    action_values[action, state] = agent.action_values[(state, action)]
                except KeyError:
                    pass
        
        greedy_actions = 1+np.argmax(action_values, axis=0)
        pertinent_states = np.concatenate([[2+k+4*i for k in range(3)] for i in range(4)])
        pertinent_actions = greedy_actions[pertinent_states]
        expected_actions = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3])
        self.assertTrue(np.all(pertinent_actions==expected_actions))
