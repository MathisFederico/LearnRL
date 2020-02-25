import unittest
import numpy as np


class EvaluationTest(unittest.TestCase):

    def setUp(self):
        from agents import Memory
        self.memory = Memory()
        self.memory.remember(0, 0, 0, False, 1)
        self.memory.remember(1, 0, 2, True, 1)
        self.action_values = {(0,0):1, (0,1):0, (1, 0):1, (2, 0):0}
        self.action_visits = {(0,0):1, (0,1):1, (1, 0):1, (2, 0):1}

    def test_TD(self):
        from agents.basic.evaluation import TemporalDifference
        td = TemporalDifference()
        td.learn(self.action_visits, self.action_values, self.memory, online=True)
        expected_action_visits = {(0, 0): 1, (0, 1): 1, (1, 0): 2, (2, 0): 1}
        expected_action_values = {(0, 0): 1, (0, 1): 0, (1, 0): 1.1, (2, 0): 0}
        expected_memory = {key:None for key in self.memory.MEMORY_KEYS}
        self.assertTrue(np.all([self.action_visits[key]==expected_action_visits[key] for key in expected_action_visits]),
                        f"An error occured in action_visits\n \
                        Expected {expected_action_visits}\n \
                        Got {self.action_visits}")
        self.assertTrue(np.all([self.action_values[key]==expected_action_values[key] for key in expected_action_values]),
                        f"An error occured in action_values\n \
                        Expected {expected_action_values}\n \
                        Got {self.action_values}")
        self.assertTrue(np.all([self.memory.datas[key]==expected_memory[key] for key in expected_memory]),
                        f"An error occured in memory\n \
                        Expected {expected_memory}\n \
                        Got {self.memory.datas}")
