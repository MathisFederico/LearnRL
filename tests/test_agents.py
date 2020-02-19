import unittest
import numpy as np

from copy import deepcopy

# General agent testing

class MemoryTest(unittest.TestCase):

    """Test the behavior of the general memory for agents"""
    
    def setUp(self):
        """Create a memory instance"""
        from agents import Memory
        self.memory = Memory()

    def test_forget(self):
        """Test that we do reset memory"""
        initial_datas = {key:None for key in self.memory.MEMORY_KEYS}
        self.memory.remember(0, 0, 0, False, 0, info={'parameter':0})
        self.assertNotEqual(self.memory.datas, initial_datas,
            'We are suppose to remember a dummy experience')
        self.memory.forget()
        for key in initial_datas:
            self.assertTrue(self.memory.datas[key] is None,
                'We are suppose to have forgotten the dummy experience \
                    \nBut in key {}\
                    \nWe still remember {}\
                    \nInstead of {}'.format(key, self.memory.datas[key], initial_datas[key]))

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

        # Test shape consistensy forced by numpy.ndarrays
        with self.assertRaises(ValueError):
            self.memory.remember(state=0, action=1.0, reward=2, done=False, next_state=3, info={'param':4})
            self.memory.remember(state=[0, 1], action=[2, 3], reward=4, done=False, next_state=[5, 6], info={'param':7})

# Basic agents testing

class ControlTest(unittest.TestCase):

    def setUp(self):
        self.action_values = np.array(range(10)[::-1])
        self.action_visits = np.array(range(10)[::-1])

    def test_getPolicy(self):
        from agents.basic.control import Control
        control = Control(name="default_policy")
        with self.assertRaises(NotImplementedError):
            control.getPolicy(action_values=None)

    def test_Greedy(self):
        from agents.basic.control import Greedy
        greedy = Greedy()
        policy = greedy.getPolicy(action_values=self.action_values)
        expected_policy = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.all(policy==expected_policy),
                        'Wrong policy for greedy\nExpected {}\nGot {}'.format(expected_policy, policy))

        greedy = Greedy(0.5)
        policy = greedy.getPolicy(action_values=self.action_values)
        expected_policy = np.array([0.55, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05])
        self.assertTrue(np.all(policy==expected_policy),
                        'Wrong policy for eps-greedy with eps=0.5\nExpected {}\nGot {}'.format(expected_policy, policy))

        greedy = Greedy(1)
        policy = greedy.getPolicy(action_values=self.action_values)
        expected_policy = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.assertTrue(np.all(policy==expected_policy),
                        'Wrong policy for eps-greedy with eps=1\nExpected {}\nGot {}'.format(expected_policy, policy))

    def test_UCB(self):
        from agents.basic.control import UCB
        ucb = UCB()

        with self.assertRaises(ValueError):
            ucb.getPolicy(action_values=None, action_visits=None)

        policy = ucb.getPolicy(action_values=self.action_values, action_visits=self.action_visits)
        expected_policy = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.all(policy==expected_policy),
                        'Wrong policy for UCB greedy\nExpected {}\nGot {}'.format(expected_policy, policy))

        ucb = UCB(2)
        policy = ucb.getPolicy(action_values=self.action_values, action_visits=self.action_visits)
        expected_policy = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.all(policy==expected_policy),
                        'Wrong policy for UCB c=2\nExpected {}\nGot {}'.format(expected_policy, policy))
        
        ucb = UCB(10)
        policy = ucb.getPolicy(action_values=self.action_values, action_visits=self.action_visits)
        expected_policy = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertTrue(np.all(policy==expected_policy),
                        'Wrong policy for UCB c=10\nExpected {}\nGot {}'.format(expected_policy, policy))

    def test_Puct(self):
        from agents.basic.control import Puct
        puct = Puct()

        with self.assertRaises(ValueError):
            puct.getPolicy(action_values=None, action_visits=None)

        policy = puct.getPolicy(action_values=self.action_values, action_visits=self.action_visits)
        expected_policy = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertTrue(np.all(policy==expected_policy),
                        'Wrong policy for puct greedy\nExpected {}\nGot {}'.format(expected_policy, policy))

        puct = Puct(2)
        policy = puct.getPolicy(action_values=self.action_values, action_visits=self.action_visits)
        expected_policy = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertTrue(np.all(policy==expected_policy),
                        'Wrong policy for puct c=2\nExpected {}\nGot {}'.format(expected_policy, policy))
        
        puct = Puct(10)
        policy = puct.getPolicy(action_values=self.action_values, action_visits=self.action_visits)
        expected_policy = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        self.assertTrue(np.all(policy==expected_policy),
                        'Wrong policy for puct c=10\nExpected {}\nGot {}'.format(expected_policy, policy))


class EvaluationTest(unittest.TestCase):

    def setUp(self):
        pass

    def test_in_dict(self):
        from agents.basic.evaluation import Evaluation

        evaluation = Evaluation(name='default')
        action_values = {(0, 0):1, (0, 1):2, (1, 0):3, (1, 1):4}
        state = np.array([0, 1, 2, 0, 1 ,2])
        action = np.array([0, 1, 0, 1, 0 ,2])
        evaluation.in_dict(action_values, state, action)

if __name__ == "__main__":
    unittest.main()
