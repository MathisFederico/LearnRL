# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from learnrl.callbacks.callback import Callback

class Metric():

    """ An helper object to represent a metric via a str metric_code """

    def __init__(self, metric_code):
        split_code = metric_code.split('.')
        self.code = metric_code
        fullname = split_code[0]
        split_name = fullname.split('~')
        self.name = split_name[0]
        self.surname = split_name[1] if len(split_name) > 1 else self.name
        self.operator = split_code[1] if len(split_code) > 1 else 'avg'
    
    def __eq__(self, other):
        if isinstance(other, str):
            return self.name == other
        return self.name == other.name
    
    def __str__(self):
        return self.name

    def __repr__(self):
        return self.code


class MetricList():

    """ An helper object to represent a list of metrics """

    def __init__(self, metric_list):
        self.metric_list = [self.to_metric(m) for m in metric_list]
        self.metric_names = [m.name for m in self.metric_list]
    
    def __getitem__(self, index):
        return self.metric_list[index]
    
    def __contains__(self, metric:Metric):
        return metric.name in self.metric_names

    def __add__(self, metric_list):
        new_list = self.metric_list + [self.to_metric(m) for m in metric_list if m not in self]
        return MetricList(new_list)

    def append(self, metric:Metric):
        if metric not in self:
            self.metric_list += [self.to_metric(metric)]
            self.metric_names += [metric.name]
        else:
            raise ValueError(f'{metric} already in MetricList')
    
    def __str__(self):
        return str(self.metric_list)
    
    @staticmethod
    def to_metric(metric):
        if isinstance(metric, Metric):
            return metric
        else:
            return Metric(metric)


class LoggingCallback(Callback):

    """ Generic class for tracking metrics """

    def __init__(self,
                detailed_step_metrics=['observation', 'action', 'next_observation'],
                step_metrics=['reward', 'loss', 'exploration~exp', 'learning_rate~lr', 'dt_step~'],
                steps_cycle_metrics=['reward', 'loss', 'exploration~exp', 'learning_rate~lr', 'dt_step~'],
                episode_only_metrics=['dt_episode~'], 
                episode_metrics=['reward.sum', 'loss', 'exploration~exp.last', 'learning_rate~lr.last', 'dt_step~'],
                episodes_cycle_only_metrics=['dt_episode~'],
                episodes_cycle_metrics=['reward', 'loss', 'exploration~exp.last', 'learning_rate~lr.last', 'dt_step~'],
                ):

        self.detailed_step_metrics = MetricList(detailed_step_metrics)

        self.step_metrics = MetricList(step_metrics)
        self.steps_cycle_metrics = MetricList(steps_cycle_metrics)

        self.episode_only_metrics = MetricList(episode_only_metrics)
        self.episode_metrics = MetricList(episode_metrics)

        self.episodes_cycle_only_metrics = MetricList(episodes_cycle_only_metrics)
        self.episodes_cycle_metrics = MetricList(episodes_cycle_metrics)

    def _reset_attr(self, attr_name, operator):
        """ Reset a metric attribute based on the metric operator """
        setattr(self, attr_name, 'N/A')
        if operator == 'avg':
            metric_seen = attr_name + '_seen'
            setattr(self, metric_seen, 0)

    def _update_attr(self, attr_name, last_value, operator):
        """ Update a metric attribute based on the metric operator and the last metric value """
        previous_value = getattr(self, attr_name)

        if previous_value == 'N/A':
            previous_value = 0

        if operator == 'avg':
            metric_seen = attr_name + '_seen'
            new_seen = getattr(self, metric_seen) + 1
            setattr(self, metric_seen, new_seen)
            setattr(self, attr_name, previous_value + (last_value - previous_value) / new_seen)

        elif operator == 'sum':
            setattr(self, attr_name, last_value + previous_value)

        elif operator == 'last':
            setattr(self, attr_name, last_value)

        else:
            raise ValueError(f'Unknowed operator {operator}')

    @staticmethod
    def _get_attr_name(prefix, metric, agent_id=None):
        """ Get the attribute name of a metric/agent couple """
        if agent_id:
            return '_'.join((prefix, "agent" + str(agent_id), metric.name))
        else:
            return '_'.join((prefix, metric.name))

    @staticmethod
    def _extract_metric_from_logs(metric_name, logs, agent_id=None):
        """ Extract the last value of a metric from logs (specified for an agent or not) """

        def _search_logs(metric_name, logs:dict):
            if logs is None or metric_name in logs:
                _logs = logs
            else:
                for _, value in logs.items():
                    if isinstance(value, dict):
                        _logs = _search_logs(metric_name, value)
                        if _logs is not None:
                            return _logs
                _logs = None
            return _logs

        _logs = None
        if agent_id is not None:
            agent_logs = logs.get(f'agent_{agent_id}')
            _logs = _search_logs(metric_name, agent_logs)
            if _logs is None:
                _logs = _search_logs(metric_name, logs)
        else:
            if metric_name in logs:
                _logs = logs
            else:
                _logs = _search_logs(metric_name, logs)

        value = _logs.get(metric_name) if _logs is not None else 'N/A'
        return value

    def _get_value(self, metric:Metric, prefix=None, agent_id=None, logs=None):
        # Search for prefix-wise attr
        if prefix is not None:
            src_name = self._get_attr_name(prefix, metric, agent_id)
            src_value = getattr(self, src_name, 'N/A')
        else:
            src_value = 'N/A'

        # If not found or no source, search in logs directly
        if src_value == 'N/A' and logs is not None:
            src_value = self._extract_metric_from_logs(metric.name, logs, agent_id)
        
        return src_value

    def _update_metrics(self, metric_list:MetricList, target_prefix, source_prefix=None, logs=None, agent_id=None, reset=False):
        """ Update the logger attributes based on a metric list """
        for metric in metric_list:
            target_name = self._get_attr_name(target_prefix, metric, agent_id)

            if reset:
                self._reset_attr(target_name, metric.operator)
                continue

            src_value = self._get_value(metric, source_prefix, agent_id, logs)

            # Update attr if a value was found
            if src_value != 'N/A':
                self._update_attr(target_name, src_value, metric.operator)
    
    def _update_metrics_all_agents(self, metric_list:MetricList, target_prefix, **kwargs):
        if self.n_agents > 1:
            for agent_id in range(self.n_agents):
                self._update_metrics(metric_list, target_prefix, agent_id=agent_id, **kwargs)
        else:
            self._update_metrics(metric_list, target_prefix, agent_id=0, **kwargs)

    def on_step_end(self, step, logs={}):
        agent_id = logs.get('agent_id') if self.n_agents > 1 else None
        self._update_metrics(self.episode_metrics, 'episode', logs=logs, agent_id=agent_id)
        if self.params['verbose'] == 3:
            self._update_metrics_all_agents(self.steps_cycle_metrics, 'steps_cycle', logs=logs)

    def on_steps_cycle_begin(self, step, logs=None):
        if self.params['verbose'] == 3:
            self._update_metrics_all_agents(self.steps_cycle_metrics, 'steps_cycle', logs=logs, reset=True)

    def on_episode_begin(self, episode, logs=None):
        self._update_metrics_all_agents(self.episode_metrics, 'episode', logs=logs, reset=True)

    def on_episode_end(self, episode, logs=None):
        if self.params['verbose'] == 1:
            self._update_metrics_all_agents(self.episodes_cycle_metrics, 'episodes_cycle', source_prefix='episode', logs=logs)

    def on_episodes_cycle_begin(self, episode, logs=None):
        if self.params['verbose'] == 1:
            self._update_metrics_all_agents(self.episodes_cycle_metrics, 'episodes_cycle', source_prefix='episode', logs=logs, reset=True)

    def on_run_begin(self, logs=None):
        self.n_agents = len(self.playground.agents)

