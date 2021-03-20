# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=protected-access, attribute-defined-outside-init, unused-argument, too-many-instance-attributes

""" Test playground.py """

import pytest
import pytest_check as check

import numpy as np

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
        " should raise a TypeError if any agent isn't a subclass of learnrl.Agent. "
        with pytest.raises(TypeError, match=r"agent.*learnrl.Agent"):
            Playground(self.env, [Agent(), 'agent'])

    def test_run(self, mocker):
        """ run should call callbacks at the right time and in the right order. """

        steps_outputs = [
            (f'obs_{i+1}', False)
            for i in range(9)
        ]
        steps_outputs += [(f'obs_{9}', True)]
        steps_outputs = steps_outputs*10
        steps_outputs = steps_outputs[::-1]

        def dummy_run_step(*args, **kwargs):
            return steps_outputs.pop()

        class RegisterCallback(Callback):

            """Dummy Callback to register calls"""

            def __init__(self):
                super().__init__()
                self.stored_key = ""

            def on_run_begin(self, logs=None):
                self.stored_key += "|-"

            def on_episodes_cycle_begin(self, episode, logs=None):
                self.stored_key += "["

            def on_episode_begin(self, episode, logs=None):
                self.stored_key += "("

            def on_steps_cycle_begin(self, step, logs=None):
                self.stored_key += "<"

            def on_step_begin(self, step, logs=None):
                self.stored_key += ","

            def on_step_end(self, step, logs=None):
                self.stored_key += "."

            def on_steps_cycle_end(self, step, logs=None):
                self.stored_key += ">"

            def on_episode_end(self, episode, logs=None):
                self.stored_key += ")"

            def on_episodes_cycle_end(self, episode, logs=None):
                self.stored_key += "]"

            def on_run_end(self, logs=None):
                self.stored_key += "-|"

        mocker.patch(
            'learnrl.playground.Playground._get_episodes_cycle_len',
            return_value=3
        )
        mocker.patch(
            'learnrl.playground.Playground._reset',
            lambda *args: ('obs_0', 0, False, {})
        )
        mocker.patch(
            'learnrl.playground.Playground._build_callbacks',
            lambda self, callbacks, logger, params: callbacks[0]
        )
        mocker.patch(
            'learnrl.playground.Playground._run_step',
            dummy_run_step
        )

        playground = Playground(self.env, self.agents)
        register_callback = RegisterCallback()
        playground.run(episodes=10, callbacks=[register_callback], steps_cycle_len=3)
        episode_key = "(<,.,.,.><,.,.,.><,.,.,.><,.>)"
        expected_key = "|-[" + episode_key*3 + "][" + episode_key*3 + "][" + \
            episode_key*3  + "][" + episode_key + "]-|"
        check.equal(register_callback.stored_key, expected_key)

    def test_fit(self, mocker):
        """ fit should call run with learn=True and render=False. """
        mocker.patch('learnrl.playground.Playground.run')
        playground = Playground(self.env, self.agents)
        playground.fit(10)
        _, kwargs = playground.run.call_args
        check.is_true(kwargs.get('learn'))
        check.is_false(kwargs.get('render'))

    def test_fit_warn_learn(self, mocker):
        """ fit should warn a UserWarning if learn=False. """
        mocker.patch('learnrl.playground.Playground.run')
        playground = Playground(self.env, self.agents)
        with pytest.warns(UserWarning, match=r".*agents will not improve.*"):
            playground.fit(10, learn=False)

    def test_fit_warn_render(self, mocker):
        """ fit should warn a RuntimeWarning if render=True. """
        mocker.patch('learnrl.playground.Playground.run')
        playground = Playground(self.env, self.agents)
        with pytest.warns(RuntimeWarning, match=r".*computation speed.*"):
            playground.fit(10, render=True)

    def test_test(self, mocker):
        """ test should call run with learn=False and render=True. """
        mocker.patch('learnrl.playground.Playground.run')
        playground = Playground(self.env, self.agents)
        playground.test(10)
        _, kwargs = playground.run.call_args
        check.is_false(kwargs.get('learn'))
        check.is_true(kwargs.get('render'))

    def test_test_warn_learn(self, mocker):
        """ test should warn a UserWarning if learn=True. """
        mocker.patch('learnrl.playground.Playground.run')
        playground = Playground(self.env, self.agents)
        with pytest.warns(UserWarning, match=r".*not act greedy.*"):
            playground.test(10, learn=True)

    def test_test_warn_render(self, mocker):
        """ test should warn a UserWarning if render=False. """
        mocker.patch('learnrl.playground.Playground.run')
        playground = Playground(self.env, self.agents)
        with pytest.warns(UserWarning, match=r".*render=True.*"):
            playground.test(10, render=False, verbose=0)

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

    def test_build_callback(self, mocker):
        """ should build a CallbackList from given callbacks and logger
        and set their params and playground. """
        callback_path = 'learnrl.callbacks.callback.Callback'
        mocker.patch(callback_path + '.set_params')
        mocker.patch(callback_path + '.set_playground')

        callbacks = [Callback(), Callback()]
        logger = Callback()
        params = {'param123': 123}
        callbacklist = self.playground._build_callbacks(callbacks, logger, params)
        check.is_instance(callbacklist, CallbackList)
        for callback in callbacklist.callbacks:
            args, _ = callback.set_params.call_args
            check.equal(args[0], params)
            args, _ = callback.set_playground.call_args
            check.equal(args[0], self.playground)
        check.is_in(logger, callbacklist.callbacks)

    def test_builb_callback_no_logger(self):
        """ should work if logger is None. """
        callbacks = [Callback(), Callback()]
        logger = None
        params = {'param123': 123}
        callbacklist = self.playground._build_callbacks(callbacks, logger, params)
        check.is_instance(callbacklist, CallbackList)
        for callback in callbacklist.callbacks:
            check.equal(callback.params, params)
            check.equal(callback.playground, self.playground)


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
        """ should return the agent designed by agent_order and turn if env is a TurnEnv. """
        mocker.patch('learnrl.envs.TurnEnv.turn', return_value=2)
        playground = Playground(TurnEnv(), self.agents)
        playground.agents_order = [3, 2, 1, 0, 4]
        _, agent_id = playground._get_next_agent('observation')
        check.equal(agent_id, 1)

    def test_turnenv_indexerror(self, mocker):
        """ should raise ValueError if turn result is out of agent_order. """
        mocker.patch('learnrl.envs.TurnEnv.turn', return_value=10)
        playground = Playground(TurnEnv(), self.agents)
        with pytest.raises(ValueError, match=r'Not enough agents.*'):
            playground._get_next_agent('observation')


class TestPlaygroundCallHandlers:
    """Playground._call_handlers"""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup of used fixtures"""
        self.call_handlers = Playground._call_handlers
        self.done = False
        self.reward = 1.2
        self.experience = {
            'observation': 'observation',
            'action': 'action',
            'reward': self.reward,
            'done': self.done,
            'next_observation': 'next_observation',
            'info': {'info': 'info'}
        }


    def test_no_handlers(self):
        """ should not log or change experience if no handler is given. """
        logs = {}
        self.call_handlers(
            self.reward,
            self.done,
            self.experience,
            reward_handler=None,
            done_handler=None,
            logs=logs
        )
        check.equal(self.experience['done'], self.done)
        check.equal(logs['done'], self.done)

        check.equal(self.experience['reward'], self.reward)
        check.equal(logs['reward'], self.reward)

    def test_done_only(self):
        """ should log and change experience correctly with a done_handler only. """
        logs = {}
        handled_done = not self.done
        self.call_handlers(
            self.reward,
            self.done,
            self.experience,
            reward_handler=None,
            done_handler=lambda *args: handled_done,
            logs=logs
        )
        check.equal(self.experience['done'], handled_done)
        check.equal(logs['done'], self.done)
        check.equal(logs['handled_done'], handled_done)

        check.equal(self.experience['reward'], self.reward)
        check.equal(logs['reward'], self.reward)

    def test_reward_only(self):
        """ should log and change experience correctly with a reward_handler only. """
        logs = {}
        handled_reward = 1.7
        self.call_handlers(
            self.reward,
            self.done,
            self.experience,
            reward_handler=lambda *args:handled_reward,
            done_handler=None,
            logs=logs
        )
        check.equal(self.experience['done'], self.done)
        check.equal(logs['done'], self.done)

        check.equal(self.experience['reward'], handled_reward)
        check.equal(logs['reward'], self.reward)
        check.equal(logs['handled_reward'], handled_reward)

    def test_both_handlers(self):
        """ should log and change experience correctly with a both handlers. """
        logs = {}
        handled_done = not self.done
        handled_reward = 1.7
        self.call_handlers(
            self.reward,
            self.done,
            self.experience,
            reward_handler=lambda *args: handled_reward,
            done_handler=lambda *args: handled_done,
            logs=logs
        )
        check.equal(self.experience['done'], handled_done)
        check.equal(logs['done'], self.done)
        check.equal(logs['handled_done'], handled_done)

        check.equal(self.experience['reward'], handled_reward)
        check.equal(logs['reward'], self.reward)
        check.equal(logs['handled_reward'], handled_reward)

class TestPlaygroundRunStep:
    """Playground._run_step"""

    @pytest.fixture(autouse=True)
    def setup_playground(self, mocker):
        """Setup of used fixtures"""

        self.observation = 'observation'
        self.next_observation = 'next_observation'
        self.reward = 1.2
        self.handled_reward = 1.7
        self.done = False
        self.info = {'env_info': 'env_info'}

        mocker.patch('gym.Env.render')
        mocker.patch('gym.Env.step',
            return_value=(self.next_observation, self.reward,
                self.done, self.info)
        )
        self.env = Env()

        mocker.patch('learnrl.agent.Agent.remember')
        mocker.patch('learnrl.agent.Agent.learn')
        self.action = 3
        mocker.patch('learnrl.agent.Agent.act', return_value=self.action)
        self.n_agents = 5
        self.agents = [Agent() for _ in range(self.n_agents)]

        self.agent_id = 0
        mocker.patch('learnrl.playground.Playground._get_next_agent',
            return_value=(self.agents[self.agent_id], self.agent_id))

        def handler_mocker(cls, reward, done, experience, reward_handler, done_handler, logs):
            experience['reward'] = self.handled_reward
            logs['handled_reward'] = self.handled_reward

        mocker.patch('learnrl.playground.Playground._call_handlers', handler_mocker)
        self.playground = Playground(self.env, self.agents)

        self.previous = [
            {'observation':None,'action':None,
            'reward':None, 'done':None, 'info':None}
            for _ in range(self.n_agents)
        ]

    def test_run_step(self):
        """ should update the observation, done and logs correcty. """
        logs = {}
        done = not self.done
        observation, done = self.playground._run_step(
            self.observation,
            self.previous,
            logs=logs,
        )
        check.equal(observation, self.next_observation)
        check.equal(done, self.done)

        for log_name in ['reward', 'handled_reward', 'observation',
                            'next_observation', 'info', 'done', 'agent_id', 'action']:
            expected = getattr(self, log_name)
            check.equal(logs[log_name], expected)

    def test_render_not_done(self, mocker):
        """ should render at each step beginning if not done. """
        render_mode = 'render_mode'
        self.playground._run_step(
            self.observation,
            self.previous,
            logs={},
            render=True,
            render_mode=render_mode
        )

        check.equal(len(self.env.render.call_args_list), 1)
        render_args, _ = self.env.render.call_args
        check.equal(render_args[0], render_mode)

    def test_render_done(self, mocker):
        """ should render at step beginning and end if done. """
        render_mode = 'render_mode'
        self.env.step.return_value = (
            self.next_observation, self.reward, True, self.info
        )
        self.playground._run_step(
            self.observation,
            self.previous,
            logs={},
            render=True,
            render_mode=render_mode
        )

        check.equal(len(self.env.render.call_args_list), 2)
        render_args, _ = self.env.render.call_args_list[-1]
        check.equal(render_args[0], render_mode)

    def test_learn_without_prev_not_done(self, mocker):
        """ should store experience without learn or remember for first experience. """
        previous = self.previous
        self.playground._run_step(
            self.observation,
            previous,
            logs={},
            learn=True,
        )
        check.equal(previous[self.agent_id],
            {name: getattr(self, name)
            for name in ['observation', 'action', 'reward', 'done', 'info']}
        )
        check.is_false(Agent.learn.called)
        check.is_false(Agent.remember.called)

    def test_learn_not_done(self, mocker):
        """ should call learn and remember once at the beginning of the step if not done. """
        logs = {}
        previous = self.previous
        previous[0] = {
            name: getattr(self, name)
            for name in ['observation', 'action', 'reward', 'done', 'info']
        }
        self.playground._run_step(
            self.observation,
            previous,
            logs=logs,
            learn=True,
        )

        check.equal(len(Agent.learn.call_args_list), 1)
        check.equal(len(Agent.remember.call_args_list), 1)
        check.is_in('agent_0', logs)

    def test_learn_done(self, mocker):
        """ should call learn and remember once at the beginning of the step if not done. """
        logs = {}
        previous = self.previous
        self.env.step.return_value = (
            self.next_observation, self.reward, True, self.info
        )
        self.playground._run_step(
            self.observation,
            previous,
            logs=logs,
            learn=True,
        )

        check.equal(len(Agent.learn.call_args_list), 1)
        check.equal(len(Agent.remember.call_args_list), 1)
        check.is_in('agent_0', logs)

class TestDoneHandler:

    """DoneHandler"""

    def test_done(self, mocker):
        """ should use `done` if the output of `done` is a bool or a bool numpy array. """
        mocker.patch("learnrl.playground.DoneHandler.done", lambda *args: True)
        handler = DoneHandler()
        check.is_true(handler._done())

        mocker.patch("learnrl.playground.DoneHandler.done", lambda *args: np.array(True))
        handler = DoneHandler()
        check.is_true(handler._done())

    def test_not_bool_done(self, mocker):
        """ should raise ValueError if the output of `done` is not a bool. """
        mocker.patch("learnrl.playground.DoneHandler.done", lambda *args: 'True')
        handler = DoneHandler()
        with pytest.raises(ValueError, match=r"Done should be bool.*"):
            handler._done()

    def test_call(self, mocker):
        """ should call done on class call. """
        mocker.patch("learnrl.playground.DoneHandler._done", return_value=True)
        handler = DoneHandler()
        check.is_true(handler())
        check.is_true(handler._done.called)


class TestRewardHandler:

    """RewardHandler"""

    def test_reward(self, mocker):
        """ should use `reward` if the output of `reward` is a float or floating numpy array. """
        mocker.patch("learnrl.playground.RewardHandler.reward", lambda *args: 1.2)
        handler = RewardHandler()
        check.equal(handler._reward(), 1.2)

        mocker.patch("learnrl.playground.RewardHandler.reward", lambda *args: np.array(1.2))
        handler = RewardHandler()
        check.equal(handler._reward(), 1.2)

    def test_not_scalar_reward(self, mocker):
        """ should raise ValueError if the output of `reward` is not a float. """
        mocker.patch("learnrl.playground.RewardHandler.reward", lambda *args: '1.2')
        handler = RewardHandler()
        with pytest.raises(ValueError, match=r"Rewards should be a float.*"):
            handler._reward()

    def test_call(self, mocker):
        """ should call reward on class call. """
        mocker.patch("learnrl.playground.RewardHandler._reward", return_value=True)
        handler = RewardHandler()
        check.is_true(handler())
        check.is_true(handler._reward.called)
