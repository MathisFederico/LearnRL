# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=protected-access, attribute-defined-outside-init, unused-argument

""" Test playground.py """

import pytest
import pytest_check as check

from gym import Env
from learnrl.agent import Agent
from learnrl.envs import TurnEnv
from learnrl.playground import Playground, RewardHandler, DoneHandler
from learnrl.callbacks.callback import CallbackList, Callback

class TestPlayground:

    """ Playground """

    @pytest.fixture(autouse=True)
    def setup_playground(self):
        """Setup of used fixtures"""
        self.env = Env()
        self.n_agents = 5
        self.agents = [Agent() for _ in range(self.n_agents)]

    def test_single_agent_argument(self):
        " should transform a single agent in a list containing itself. "
        single_agent = Agent()
        playground = Playground(self.env, single_agent)
        check.equal(playground.agents, [single_agent])

    def test_env_typeerror(self):
        " should raise a TypeError if the environment isn't a subclass of gym. "
        with pytest.raises(TypeError, match=r"environement.*gym.Env"):
            Playground("env", self.agents)

    def test_agent_typeerror(self):
        " should raise a TypeError if the any agent isn't a subclass of learnrl.Agent. "
        with pytest.raises(TypeError, match=r"agent.*learnrl.Agent"):
            Playground(self.env, [Agent(), 'agent'])


class TestPlaygroundGetEpisodeCycleLen:

    """ Playground._get_episodes_cycle_len """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup of used fixtures"""
        self.get_episodes_cycle_len = Playground._get_episodes_cycle_len

    def test_over_one(self):
        " should be the floor if > 1. "
        value = self.get_episodes_cycle_len(10, 100)
        check.equal(value, 10)

        value = self.get_episodes_cycle_len(12.4, 100)
        check.equal(value, 12)

    def test_between_zero_one(self):
        " should be a proportion of total episodes if between 0 and 1. "
        value = self.get_episodes_cycle_len(0.42, 100)
        check.equal(value, 42)

    def test_bellow_zero(self):
        " should raise a ValueError if < 0. "
        with pytest.raises(ValueError, match=r".*> 0.*"):
            self.get_episodes_cycle_len(0, 100)

        with pytest.raises(ValueError, match=r".*> 0.*"):
            self.get_episodes_cycle_len(-0.2, 100)


class TestPlaygroundBuildCallbacks:
    """ Playground._build_callbacks """

    @pytest.fixture(autouse=True)
    def setup_playground(self):
        """Setup of used fixtures"""
        self.env = Env()
        self.n_agents = 5
        self.agents = [Agent() for _ in range(self.n_agents)]
        self.playground = Playground(self.env, self.agents)

    def test_builb_callback(self, mocker):
        """ should build a CallbackList from given callbacks and logger
        and set their params and playground. """
        callbacks = [Callback(), Callback()]
        logger = Callback()
        params = {'param123': 123}
        callbacklist = self.playground._build_callbacks(callbacks, logger, params)
        check.is_instance(callbacklist, CallbackList)
        for callback in callbacklist.callbacks:
            check.equal(callback.params, params)
            check.equal(callback.playground, self.playground)
        check.is_in(logger, callbacklist.callbacks)


class TestPlaygroundReset:

    """ Playground._reset """

    @pytest.fixture(autouse=True)
    def setup_playground(self):
        """Setup of used fixtures"""
        self.env = Env()
        self.n_agents = 5
        self.agents = [Agent() for _ in range(self.n_agents)]
        self.playground = Playground(self.env, self.agents)

    def test_no_handlers(self, mocker):
        """ should reset the environment with no handlers. """

        mocker.patch('gym.Env.reset', lambda self:'obs')

        observation, step, done, previous = self.playground._reset(None, None)
        check.equal(observation, 'obs')
        check.equal(step, 0)
        check.equal(done, False)
        expected_previous = [
            {'observation':None, 'action':None, 'reward':None, 'done':None, 'info':None}
            for _ in range(len(self.agents))
        ]
        check.equal(previous, expected_previous)

    def test_callable_handlers(self, mocker):
        """ should reset the environment with callable handlers. """

        def done_handler(**kwargs):
            return not kwargs.get('done')

        def reward_handler(**kwargs):
            return 2 * kwargs.get('reward')

        mocker.patch('gym.Env.reset', lambda self:'obs')

        observation, step, done, previous = self.playground._reset(reward_handler, done_handler)
        check.equal(observation, 'obs')
        check.equal(step, 0)
        check.equal(done, False)
        expected_previous = [
            {'observation':None, 'action':None, 'reward':None, 'done':None, 'info':None}
            for _ in range(len(self.agents))
        ]
        check.equal(previous, expected_previous)

    def test_with_handlers(self, mocker):
        """ should reset the environment and handlers with true handlers. """

        mocker.patch('gym.Env.reset', lambda self:'obs')

        mocker.patch('learnrl.playground.RewardHandler.reset')
        reward_handler = RewardHandler()

        mocker.patch('learnrl.playground.DoneHandler.reset')
        done_handler = DoneHandler()

        observation, step, done, previous = self.playground._reset(reward_handler, done_handler)

        check.equal(observation, 'obs')
        check.equal(step, 0)
        check.equal(done, False)
        expected_previous = [
            {'observation':None, 'action':None, 'reward':None, 'done':None, 'info':None}
            for _ in range(len(self.agents))
        ]
        check.equal(previous, expected_previous)

        check.is_true(reward_handler.reset.called)
        check.is_true(done_handler.reset.called)


class TestPlaygroundGetNextAgent:
    """ Playground._get_next_agent """

    @pytest.fixture(autouse=True)
    def setup_playground(self):
        """Setup of used fixtures"""
        self.n_agents = 5
        self.agents = [Agent() for _ in range(self.n_agents)]

    def test_no_turnenv(self):
        """ should return the first agent if env is not a TurnEnv. """
        playground = Playground(Env(), self.agents)
        _, agent_id = playground._get_next_agent('observation')
        check.equal(agent_id, 0)

    def test_turnenv(self, mocker):
        """ should return the first agent if env is not a TurnEnv. """
        mocker.patch('learnrl.envs.TurnEnv.turn', return_value=2)
        playground = Playground(TurnEnv(), self.agents)
        _, agent_id = playground._get_next_agent('observation')
        check.equal(agent_id, 2)

    def test_turnenv_indexerror(self, mocker):
        """ should return the first agent if env is not a TurnEnv. """
        mocker.patch('learnrl.envs.TurnEnv.turn', return_value=10)
        playground = Playground(TurnEnv(), self.agents)
        with pytest.raises(ValueError, match=r'Not enough agents.*'):
            playground._get_next_agent('observation')


class TestPlaygroundAgentOrder:

    """ Playground.set_agent_order """

    @pytest.fixture(autouse=True)
    def setup_playground(self):
        """Setup of used fixtures"""
        self.env = Env()
        self.n_agents = 5
        self.agents = [Agent() for _ in range(self.n_agents)]
        self.playground = Playground(self.env, self.agents)

    def test_default(self):
        """ should have a correct default agent order. """
        check.equal(self.playground.agents_order, list(range(self.n_agents)),
            f"Default agents_order shoud be {list(range(self.n_agents))}"
            f"but was {self.playground.agents_order}"
        )

    def test_custom_at_init(self):
        """ should be able to have custom order at initialization. """
        custom_order = [4, 3, 1, 2, 0]
        playground = Playground(self.env, self.agents, agents_order=custom_order)
        check.equal(playground.agents_order, custom_order,
            f"Custom agents_order shoud be {custom_order}"
            f"but was {playground.agents_order}"
        )

    def test_custom_after_init(self):
        """ should be able to set custom order after initialization. """
        new_custom_order = [3, 4, 1, 2, 0]
        self.playground.set_agents_order(new_custom_order)

        check.equal(self.playground.agents_order, new_custom_order,
            f"Custom agents_order shoud be {new_custom_order}"
            f"but was {self.playground.agents_order}"
        )

    def test_not_enought_indexes(self):
        """ should raise ValueError if not enough indexes in custom order. """
        with pytest.raises(ValueError, match=r"Not every agents*"):
            Playground(self.env, self.agents, agents_order=[4, 3, 1, 2])

    def test_missing_indexes(self):
        """ should raise ValueError if missing indexes in custom order. """
        with pytest.raises(ValueError, match=r".*not taking every index*"):
            Playground(self.env, self.agents, agents_order=[4, 6, 1, 2, 0])
