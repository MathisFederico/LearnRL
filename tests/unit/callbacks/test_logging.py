# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>
# pylint: disable=protected-access, attribute-defined-outside-init, unused-argument, missing-function-docstring

""" Test metrics logging behavior """

import pytest_check as check
import pytest

from learnrl.callbacks.logging_callback import LoggingCallback, Metric, MetricList

class TestMetric:
    """ Metric """

    def test_init_full(self):
        """ should instanciate correctly with a full metric code. """
        metric = Metric('reward~rwd.sum')
        check.equal(metric.name, 'reward',
            f'Metric name should be reward and not {metric.name}'
        )
        check.equal(metric.surname, 'rwd',
            f'Metric surname should be rwd and not {metric.surname}'
        )
        check.equal(metric.operator, 'sum',
            f'Metric opertor should be sum and not {metric.operator}'
        )


    def test_init_no_op(self):
        """ should instanciate correctly without an operator. """
        metric = Metric('reward~rwd')
        check.equal(metric.name, 'reward',
            f'Metric name should be reward and not {metric.name}'
        )
        check.equal(metric.surname, 'rwd',
            f'Metric surname should be rwd and not {metric.surname}'
        )
        check.equal(metric.operator, 'avg',
            f'Metric opertor should be avg and not {metric.operator}'
        )

    def test_init_no_surname(self):
        """ should instanciate correctly without a surname. """
        metric = Metric('reward.sum')
        check.equal(metric.name, 'reward',
            f'Metric name should be reward and not {metric.name}'
        )
        check.equal(metric.surname, 'reward',
            f'Metric surname should be reward and not {metric.surname}'
        )
        check.equal(metric.operator, 'sum',
            f'Metric opertor should be sum and not {metric.operator}'
        )

    def test_init_name_only(self):
        """ should instanciate correctly with a name only. """
        metric = Metric('reward')
        check.equal(metric.name, 'reward',
            f'Metric name should be reward and not {metric.name}'
        )
        check.equal(metric.surname, 'reward',
            f'Metric surname should be reward and not {metric.surname}'
        )
        check.equal(metric.operator, 'avg',
            f'Metric opertor should be avg and not {metric.operator}'
        )

    def test_equal_str(self):
        """ should be equal to a string equal to its name only. """
        metric = Metric('reward~rwd.sum')
        check.is_true(metric == 'reward')
        check.is_false(metric == 'rewards')
        check.is_false(metric == 'rwd')

    def test_equal_metric(self):
        """ should be equal to a metric with same name. """
        metric_sum = Metric('reward~rwd.sum')
        metric_avg = Metric('reward~R.avg')
        check.is_true(metric_sum == metric_avg)

    def test_str(self):
        """ should be represented as name with str. """
        metric = Metric('reward~rwd.sum')
        check.is_true(str(metric) == 'reward')

    def test_repr(self):
        """ should be represented as full metric code with repr. """
        metric = Metric('reward~rwd.sum')
        check.is_true(repr(metric) == 'reward~rwd.sum')


class TestMetricList:
    """ MetricList """

    @pytest.fixture(autouse=True)
    def setup_metrics(self):
        """ Setup metrics for tests """
        self.metric_codes = ['reward~rwd.sum', 'loss_1', 'loss_2.sum']
        self.metrics = [Metric(code) for code in self.metric_codes]

    def test_init_metrics(self):
        """ should instanciate correclty with a list of Metric. """
        metriclist = MetricList(self.metrics)
        check.equal(metriclist.metric_list, self.metrics)
        check.equal(metriclist.metric_names, [metric.name for metric in self.metrics])

    def test_init_codes(self):
        """ should instanciate correclty with a list of metric codes. """
        metriclist = MetricList(self.metrics)
        check.equal(metriclist.metric_list, self.metrics)
        check.equal(metriclist.metric_names, [metric.name for metric in self.metrics])

    def test_str(self):
        """ should be represented correctly to str. """
        metriclist = MetricList(self.metrics)
        check.equal(str(metriclist), str(self.metrics))

    def test_add_metric(self):
        """ should add correclty a new metric. """
        metriclist = MetricList(self.metrics)
        metric = Metric('exploration~exp.last')
        metriclist += metric
        check.equal(metriclist.metric_names, self.metrics + [metric])
        expected_metric_names = [metric.name for metric in self.metrics] + ['exploration']
        check.equal(metriclist.metric_names, expected_metric_names)

    def test_add_metric_code(self):
        """ should add correclty a new metric code. """
        metriclist = MetricList(self.metrics)
        metric = Metric('exploration~exp.last')
        metriclist += 'exploration~exp.last'
        check.equal(metriclist.metric_names, self.metrics + [metric])
        expected_metric_names = [metric.name for metric in self.metrics] + ['exploration']
        check.equal(metriclist.metric_names, expected_metric_names)

    def test_add_metric_list(self):
        """ should add correclty a list of Metric. """
        metriclist = MetricList(self.metrics)
        exploration = Metric('exploration~exp.last')
        decay = Metric('decay.last')
        metriclist += [exploration, decay]
        check.equal(metriclist.metric_names, self.metrics + [exploration, decay])
        expected_metric_names = [metric.name for metric in self.metrics] + ['exploration', 'decay']
        check.equal(metriclist.metric_names, expected_metric_names)

    def test_add_metric_codes_list(self):
        """ should add correclty a new list of metric codes. """
        metriclist = MetricList(self.metrics)
        exploration = Metric('exploration~exp.last')
        decay = Metric('decay.last')
        metriclist += ['exploration~exp.last', 'decay.last']
        check.equal(metriclist.metric_names, self.metrics + [exploration, decay])
        expected_metric_names = [metric.name for metric in self.metrics] + ['exploration', 'decay']
        check.equal(metriclist.metric_names, expected_metric_names)

    def test_add_metriclist(self):
        """ should concatenate correclty two MetricList. """
        metriclist = MetricList(self.metrics)
        exploration = Metric('exploration~exp.last')
        decay = Metric('decay.last')
        metriclist += MetricList([exploration, decay])
        check.equal(metriclist.metric_names, self.metrics + [exploration, decay])
        expected_metric_names = [metric.name for metric in self.metrics] + ['exploration', 'decay']
        check.equal(metriclist.metric_names, expected_metric_names)


class TestLoggingCallback:
    """ LoggingCallback """

    @pytest.fixture(autouse=True)
    def setup(self):
        """ Setup fixtures. """
        self.logging_callback_path = "learnrl.callbacks.logging_callback.LoggingCallback"

    def test_init(self):
        """ should instanciate correctly. """
        LoggingCallback()

    def test_reset_attr_avg(self):
        """ should reset arguments correctly. """
        logging_callback = LoggingCallback()
        logging_callback.reward = 0
        logging_callback._reset_attr('reward', 'avg')
        check.equal(logging_callback.reward, 'N/A')
        check.equal(logging_callback.reward_seen, 0)

    def test_reset_attr_anyop(self):
        """ should reset arguments correctly. """
        logging_callback = LoggingCallback()
        logging_callback.reward = 0
        logging_callback._reset_attr('reward', 'x')
        check.equal(logging_callback.reward, 'N/A')

    def test_update_attr_avg(self):
        """ should update arguments correctly with avg operator. """
        logging_callback = LoggingCallback()
        logging_callback.reward = 0
        logging_callback.reward_seen = 2
        logging_callback._update_attr('reward', 1, 'avg')
        check.equal(logging_callback.reward, 1/3)
        check.equal(logging_callback.reward_seen, 3)

        # With N/A
        logging_callback = LoggingCallback()
        logging_callback.reward = 'N/A'
        logging_callback.reward_seen = 0
        logging_callback._update_attr('reward', 1, 'avg')
        check.equal(logging_callback.reward, 1)
        check.equal(logging_callback.reward_seen, 1)

    def test_update_attr_sum(self):
        """ should update arguments correctly with sum operator. """
        logging_callback = LoggingCallback()
        logging_callback.reward = 2
        logging_callback._update_attr('reward', 1, 'sum')
        check.equal(logging_callback.reward, 3)

    def test_update_attr_last(self):
        """ should update arguments correctly with last operator. """
        logging_callback = LoggingCallback()
        logging_callback.reward = 2
        logging_callback._update_attr('reward', 1, 'last')
        check.equal(logging_callback.reward, 1)

    def test_update_attr_raise(self):
        """ should raise ValueError if operator is unknowed. """
        with pytest.raises(ValueError, match=r"Unknowed operator.*"):
            logging_callback = LoggingCallback()
            logging_callback.reward = 'N/A'
            logging_callback._update_attr('reward', 1, 'x')

    def test_on_run_begin(self, mocker):
        """ should set n_agents on_run_begin. """
        class DummyPlayground():
            """DummyPlaygrounf"""
            def __init__(self, n_agents):
                self.agents = [None] * n_agents

        n_agents = 5
        mocker.patch(self.logging_callback_path + '', return_value=n_agents)
        logging_callback = LoggingCallback()
        logging_callback.playground = DummyPlayground(n_agents)
        check.is_none(logging_callback.n_agents)
        logging_callback.on_run_begin()
        check.equal(logging_callback.n_agents, n_agents)

    def test_on_episodes_cycle_begin(self, mocker):
        """ should reset episodes_cycle_metrics on_episodes_cycle_begin. """
        mocker.patch(self.logging_callback_path + '._update_metrics_all_agents')
        logging_callback = LoggingCallback()
        logging_callback.on_episodes_cycle_begin(episode=7)
        args, kwargs = logging_callback._update_metrics_all_agents.call_args
        check.equal(args[1], 'episodes_cycle')
        check.is_true(kwargs.get('reset'))

    def test_on_episode_begin(self, mocker):
        """ should reset episode_metrics on_episode_begin. """
        mocker.patch(self.logging_callback_path + '._update_metrics_all_agents')
        logging_callback = LoggingCallback()
        logging_callback.on_episode_begin(episode=7)
        args, kwargs = logging_callback._update_metrics_all_agents.call_args
        check.equal(args[1], 'episode')
        check.is_true(kwargs.get('reset'))

    def test_on_step_cycle_begin(self, mocker):
        """ should reset steps_cycle_metrics on step_cycle_begin. """
        mocker.patch(self.logging_callback_path + '._update_metrics_all_agents')
        logging_callback = LoggingCallback()
        logging_callback.on_steps_cycle_begin(step=7)
        args, kwargs = logging_callback._update_metrics_all_agents.call_args
        check.equal(args[1], 'steps_cycle')
        check.is_true(kwargs.get('reset'))

    def test_on_episode_end(self, mocker):
        """ should update episodes_cycle_metrics on_episode_end from episode metrics. """
        mocker.patch(self.logging_callback_path + '._update_metrics_all_agents')
        logging_callback = LoggingCallback()
        logging_callback.on_episode_end(episode=7)
        args, kwargs = logging_callback._update_metrics_all_agents.call_args
        check.equal(args[1], 'episodes_cycle')
        check.is_false(kwargs.get('reset'))
        check.equal(kwargs.get('source_prefix'), 'episode')

    def test_on_step_end(self, mocker):
        """ should update episode_metrics for the current agent and
            steps_cycle_metrics for all agents on_step_end. """
        mocker.patch(self.logging_callback_path + '._update_metrics_all_agents')
        mocker.patch(self.logging_callback_path + '._update_metrics')

        logging_callback = LoggingCallback()
        logging_callback.n_agents = 1
        logging_callback.on_step_end(step=7)

        args, kwargs = logging_callback._update_metrics.call_args
        check.equal(args[1], 'episode')
        check.is_false(kwargs.get('reset'))
        check.is_none(kwargs.get('agent_id'))

        args, kwargs = logging_callback._update_metrics_all_agents.call_args
        check.equal(args[1], 'steps_cycle')
        check.is_false(kwargs.get('reset'))

        # Multi agents
        logging_callback = LoggingCallback()
        logging_callback.n_agents = 5
        logging_callback.on_step_end(step=7, logs={'agent_id': 3})

        args, kwargs = logging_callback._update_metrics.call_args
        check.equal(args[1], 'episode')
        check.is_false(kwargs.get('reset'))
        check.equal(kwargs.get('agent_id'), 3)

        args, kwargs = logging_callback._update_metrics_all_agents.call_args
        check.equal(args[1], 'steps_cycle')
        check.is_false(kwargs.get('reset'))


class TestGetAttrName:
    """ LoggingCallback._get_attr_name """

    @pytest.fixture(autouse=True)
    def setup(self):
        """ Retrieve static function to test. """
        self.get_attr_name = LoggingCallback._get_attr_name

    def test_without_agent(self):
        """ should name attrs correctly. """
        metric = Metric('reward~rwd')
        name = self.get_attr_name('prefix', metric, agent_id=None)
        expected_name = 'prefix_reward'
        check.equal(name, expected_name)

    def test_with_specific_agent(self):
        """ should name attrs correctly with specific agent. """
        metric = Metric('reward~rwd')
        name = self.get_attr_name('prefix', metric, agent_id=2)
        expected_name = 'prefix_agent2_reward'
        check.equal(name, expected_name)


class TestExtractFromLogs:
    """ LoggingCallback._extract_from_logs """

    @pytest.fixture(autouse=True)
    def setup(self):
        """ Setup fake logs and retrieve function to test. """
        self.extract_from_logs = LoggingCallback._extract_metric_from_logs
        self.logs = {
            'value': 0,
            'step': 2,
            'agent_0': {
                'value': 1,
            },
            'agent_1': {
                'value': 2,
                'specific_value': 42,
            }
        }

    def test_find_value(self):
        """ should find value in logs. """
        value = self.extract_from_logs('value', self.logs)
        check.equal(value, 0)

    def test_find_nothing(self):
        """ should return N/A when there is no value. """
        nothing = self.extract_from_logs('nothing', self.logs)
        check.equal(nothing, 'N/A')

    def test_find_any_specific_value(self):
        """ should find any specific value in agent when no agent is specified. """
        specific_value = self.extract_from_logs('specific_value', self.logs)
        check.equal(specific_value, 42)

    def test_find_agent_specific_value(self):
        """ should find specific agent values. """
        value_0 = self.extract_from_logs('value', self.logs, agent_id=0)
        value_1 = self.extract_from_logs('value', self.logs, agent_id=1)

        check.equal(value_0, 1)
        check.equal(value_1, 2)

    def test_return_outer_value_when_no_specific_agent_value(self):
        """ should return outer value if no specific value is found when an agent is specified. """
        step = self.extract_from_logs('step', self.logs, agent_id=0)
        check.equal(step, 2)


class TestLoggingCallbackExtractLists:
    """ LoggingCallback._extract_lists """

    @pytest.fixture(autouse=True)
    def setup_logs(self):
        """ Setup fake logs and retrieve function to test. """
        self.extract_lists = LoggingCallback._extract_lists

    def test_extract_lists_use_case(self):
        """ should extract the correct metrics list. """

        metrics = [
            ('reward~rwd', {'steps': 'sum', 'episode': 'sum'}),
            ('loss', {'episodes': 'last'}),
            'exploration~exp.last',
            'decay'
        ]

        metric_lists = self.extract_lists(metrics)

        expected_metric_lists = [
            ['reward~rwd', 'loss', 'exploration~exp', 'decay'],
            ['reward~rwd.sum', 'loss.avg', 'exploration~exp.last', 'decay.avg'],
            ['reward~rwd.sum', 'loss.avg', 'exploration~exp.last', 'decay.avg'],
            ['reward~rwd.avg', 'loss.last', 'exploration~exp.last', 'decay.avg']
        ]

        metric_lists_names = [
            'step_metrics',
            'steps_cycle_metrics',
            'episode_metrics',
            'episodes_cycle_metrics'
        ]

        iterator = zip(metric_lists, expected_metric_lists, metric_lists_names)
        for metric_list, expected_metric_list, metric_list_name in iterator:
            check.equal(metric_list, expected_metric_list,
                f'Unexpected metric list got {metric_list} ' \
                f'instead of {expected_metric_list} for {metric_list_name}'
            )

    def test_extract_lists_wrong_format(self):
        """ should raise ValueError on wrong metric nested format. """
        metrics = ('reward')

        with pytest.raises(ValueError, match=r".*metrics format.*"):
            self.extract_lists(metrics)


class TestLoggingCallbackGetValue:
    """ LoggingCallback._get_value """

    def test_get_value_with_prefix(self, mocker):
        """ should return attr value when prefix is given. """
        expected_value = 123

        def _get_attr_name(*args):
            return 'attribute'

        mocker.patch(
            'learnrl.callbacks.logging_callback.LoggingCallback._get_attr_name',
            _get_attr_name
        )

        logging_callback = LoggingCallback()
        logging_callback.attribute = expected_value

        metric = Metric('attribute')
        value = logging_callback._get_value(metric, prefix='prefix')

        check.equal(value, expected_value)

    def test_get_value_no_prefix_logs(self, mocker):
        """ should return value in logs when no prefix is given. """
        expected_value = 123

        def _extract_metric_from_logs(self, metric_name, logs, agent_id):
            return logs[metric_name]

        mocker.patch(
            'learnrl.callbacks.logging_callback.LoggingCallback._extract_metric_from_logs',
            _extract_metric_from_logs
        )

        logging_callback = LoggingCallback()
        logs = {'attribute': 123}

        metric = Metric('attribute')
        value = logging_callback._get_value(metric, logs=logs)

        check.equal(value, expected_value)

    def test_get_value_no_prefix_no_logs(self, mocker):
        """ should return N/A when nothing is found. """
        expected_value = 'N/A'
        logging_callback = LoggingCallback()
        metric = Metric('attribute')
        value = logging_callback._get_value(metric)
        check.equal(value, expected_value)


class TestLoggingCallbackUpdateMetrics:
    """ LoggingCallback._update_metrics """

    @pytest.fixture(autouse=True)
    def setup(self):
        self.metric_list = MetricList([
            Metric("metric_1"),
            Metric("metric_2"),
            Metric("metric_2")
        ])

        self.logging_callback = LoggingCallback()

    def test_na_value(self, mocker):
        """ should not update attr if value is N/A. """
        logging_callback_path = 'learnrl.callbacks.logging_callback.LoggingCallback'
        mocker.patch(logging_callback_path + '._get_attr_name', return_value="target_name")
        mocker.patch(logging_callback_path + '._reset_attr')
        mocker.patch(logging_callback_path + '._get_value', return_value='N/A')
        mocker.patch(logging_callback_path + '._update_attr')

        self.logging_callback._update_metrics(
            self.metric_list,
            'target_prefix',
            source_prefix='source_prefix',
            logs='logs',
            agent_id='agent_id',
            reset=False
        )

        for args, _ in self.logging_callback._get_attr_name.call_args_list:
            check.equal(args[0], 'target_prefix')
            check.equal(args[2], 'agent_id')

        for args, _ in self.logging_callback._get_value.call_args_list:
            check.equal(args[1], 'source_prefix')
            check.equal(args[2], 'agent_id')
            check.equal(args[3], 'logs')

        check.is_false(self.logging_callback._reset_attr.called)
        check.is_false(self.logging_callback._update_attr.called)

    def test_update(self, mocker):
        """ should update attr if value is not N/A. """
        logging_callback_path = 'learnrl.callbacks.logging_callback.LoggingCallback'
        mocker.patch(logging_callback_path + '._get_attr_name', return_value="target_name")
        mocker.patch(logging_callback_path + '._reset_attr')
        mocker.patch(logging_callback_path + '._get_value', return_value="value")
        mocker.patch(logging_callback_path + '._update_attr')

        self.logging_callback._update_metrics(
            self.metric_list,
            'target_prefix',
            source_prefix='source_prefix',
            logs='logs',
            agent_id='agent_id',
            reset=False
        )

        for args, _ in self.logging_callback._get_attr_name.call_args_list:
            check.equal(args[0], 'target_prefix')
            check.equal(args[2], 'agent_id')

        for args, _ in self.logging_callback._get_value.call_args_list:
            check.equal(args[1], 'source_prefix')
            check.equal(args[2], 'agent_id')
            check.equal(args[3], 'logs')

        check.is_false(self.logging_callback._reset_attr.called)

        for args, _ in self.logging_callback._update_attr.call_args_list:
            check.equal(args[0], 'target_name')
            check.equal(args[1], 'value')

    def test_reset(self, mocker):
        """ should reset attr if reset is True. """
        logging_callback_path = 'learnrl.callbacks.logging_callback.LoggingCallback'
        mocker.patch(logging_callback_path + '._get_attr_name', return_value="target_name")
        mocker.patch(logging_callback_path + '._reset_attr')
        mocker.patch(logging_callback_path + '._get_value', return_value="value")
        mocker.patch(logging_callback_path + '._update_attr')

        self.logging_callback._update_metrics(
            self.metric_list,
            'target_prefix',
            source_prefix='source_prefix',
            logs='logs',
            agent_id='agent_id',
            reset=True
        )

        for args, _ in self.logging_callback._get_attr_name.call_args_list:
            check.equal(args[0], 'target_prefix')
            check.equal(args[2], 'agent_id')

        for args, _ in self.logging_callback._get_value.call_args_list:
            check.equal(args[1], 'source_prefix')
            check.equal(args[2], 'agent_id')
            check.equal(args[3], 'logs')

        for args, _ in self.logging_callback._reset_attr.call_args_list:
            check.equal(args[0], 'target_name')

        check.is_false(self.logging_callback._update_attr.called)

    def test_all_agents(self, mocker):
        """ should update all agents indepentently. """
        logging_callback_path = 'learnrl.callbacks.logging_callback.LoggingCallback'
        mocker.patch(logging_callback_path + '._update_metrics')

        n_agents = 5
        self.logging_callback.n_agents = n_agents
        self.logging_callback._update_metrics_all_agents(
            self.metric_list,
            'target_prefix'
        )

        for i in range(n_agents):
            _, kwargs = self.logging_callback._update_metrics.call_args_list[i]
            check.equal(kwargs.get('agent_id'), i)
