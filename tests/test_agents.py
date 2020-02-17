import unittest
import numpy as np

from unittest import TestCase
from copy import deepcopy
from agents import Agent, Memory

class MemoryTest(TestCase):

    """Test the behavior of the general memory for agents"""
    
    def setUp(self):
        """Create a memory instance"""
        self.memory = Memory()

    def test_forget(self):
        """Test that we do reset memory"""
        initial_datas = deepcopy(self.memory.datas)
        self.memory.remember(0, 0, 0, False, 0, info={'parameter':0})
        self.assertNotEqual(self.memory.datas, initial_datas, 'We are suppose to remember a dummy experience')
        self.memory.forget()
        self.assertEqual(self.memory.datas, initial_datas, 'We are suppose to have forgotten the dummy experience')

    def test_remember(self):
        """
        As a reminder, we ought to remember at least those MEMORY_KEYS : (state, action, reward, done, next_state, info)
        For state, action, next_state :
            Test that we can remember int, floats, lists, or numpy.ndarrays as numpy.ndarrays
        For reward :
            Test that we can remember int, floats as numpy.ndarrays
        For done :
            Test that we can remember bool as numpy.ndarrays
        For info :
            Test that we can remember dict as a 2D numpy.ndarrays
        Test that for a same key in MEMORY_KEYS, datas have consitant shapes to create a numpy.ndarrays (except info)
        """

        # Test int and float behavior
        self.memory.remember(state=0, action=1.0, reward=2, done=False, next_state=3, info={'param':4})
        self.memory.remember(state=5, action=6.0, reward=7, done=True, next_state=8, info={'param':9})
        expected = {'state':np.array([0, 5]), 'action':np.array([1.0, 6.0]), 'reward':np.array([2, 7]),
                    'done':np.array([0, 1]), 'next_state':np.array([3, 8]), 'info':np.array([{'param':4}, {'param':9}])}
        for key in self.memory.MEMORY_KEYS:
            self.assertTrue(np.all(self.memory.datas[key]==expected[key]),
                            'Couldn\'t remember the int&float dummy exemple !\
                            \nKey : {}\nGot \n{}\ninstead of\n{}'.format(key, self.memory.datas[key], expected[key]))
        self.memory.forget()

        # Test list behavior
        self.memory.remember(state=[0, 1], action=[2, 3], reward=4, done=False, next_state=[5, 6], info={'param':7})
        self.memory.remember(state=[8, 9], action=[10, 11], reward=12, done=True, next_state=[13, 14], info={'param':15})
        expected = {'state':np.array([[0, 1], [8, 9]]), 'action':np.array([[2, 3], [10, 11]]), 'reward':np.array([4, 12]),
                    'done':np.array([0, 1]), 'next_state':np.array([[5, 6], [13, 14]]), 'info':np.array([{'param':7}, {'param':15}])}
        for key in self.memory.MEMORY_KEYS:
            self.assertTrue(np.all(self.memory.datas[key]==expected[key]),
                            'Couldn\'t remember the list dummy exemple !\
                            \n Key : {}, Got {} instead of {}'.format(key, self.memory.datas[key], expected[key]))
        self.memory.forget()

if __name__ == "__main__":
    unittest.main()