import unittest
import numpy as np


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

if __name__ == "__main__":
    unittest.main()