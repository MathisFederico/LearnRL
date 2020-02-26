import unittest
import numpy as np


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
