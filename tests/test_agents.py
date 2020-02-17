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
            Other types should raise an error
        For reward :
            Test that we can remember int, floats as numpy.ndarrays
            Other types should raise an error
        For done :
            Test that we can remember bool as numpy.ndarrays
            Other types should raise an error
        For info :
            Test that we can remember dict as a 2D numpy.ndarrays
            Other types should raise an error
        Test that for a same key in MEMORY_KEYS, datas have consitant shapes to create a numpy.ndarrays (except info)
        """

        # Test int behavior
        self.memory.remember(state=0, action=0, reward=0, done=False, next_state=0, info={'param':0})
        self.memory.remember(state=1, action=1, reward=1, done=True, next_state=1, info={'param':1})
        for key in self.memory.MEMORY_KEYS:
            if key in ('state', 'action', 'reward', 'done', 'next_state'):
                expected = np.array([0, 1])
                self.assertTrue(np.all(self.memory.datas[key]==expected),
                                'Couldn\'t remember the int dummy exemple !\
                                \n Key : {}, Got {} instead of {}'.format(key, self.memory.datas[key], expected))
        self.memory.forget()

        # Test list behavior
        self.memory.remember(state=[0, 1], action=[0, 1], reward=0, done=False, next_state=[0, 1], info={'param':1})
        self.memory.remember(state=[2, 3], action=[2, 3], reward=1, done=True, next_state=[2, 3], info={'param':1})
        for key in self.memory.MEMORY_KEYS:
            if key in ('state', 'action', 'next_state'):
                expected = np.array([[0, 1],[2, 3]])
                self.assertTrue(np.all(self.memory.datas[key]==expected),
                                'Couldn\'t remember the list dummy exemple !\
                                \n Key : {}, Got {} instead of {}'.format(key, self.memory.datas[key], expected))
            elif key in ('reward', 'done'):
                expected = np.array([0, 1])
                self.assertTrue(np.all(self.memory.datas[key]==expected),
                                'Couldn\'t remember the list dummy exemple !\
                                \n Key : {}, Got {} instead of {}'.format(key, self.memory.datas[key], expected))
        self.memory.forget()

unittest.main()