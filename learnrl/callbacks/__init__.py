import time
import sys
import numpy as np

class Callback():

    """ An object to call functions while the :class:`~learnrl.core.Playground` is running.

    You can define the custom functions `on_{position}` where position can be :
    
    >>> run_begin
    ...     cycle_begin
    ...         episode_begin
    ...             step_begin
    ...             # env.step()
    ...             step_end
    ...         # done==True
    ...         episode_end
    ...     cycle_end
    ... run_end
    
    """

    def set_params(self, params):
        self.params = params
    
    def set_playground(self, playground):
        self.playground = playground

    def on_step_begin(self, step, logs=None):
        pass

    def on_step_end(self, step, logs=None):
        pass
    
    def on_episode_begin(self, episode, logs=None):
        pass

    def on_episode_end(self, episode, logs=None):
        pass

    def on_cycle_begin(self, episode, logs=None):
        pass

    def on_cycle_end(self, episode, logs=None):
        pass

    def on_run_begin(self, logs=None):
        pass

    def on_run_end(self, logs=None):
        pass

class CallbackList():
    """ An wrapper to use a list of :class:`Callback` while the :class:`~learnrl.core.Playground` is running.
    """
    
    def __init__(self, callbacks=[]):
        self.callbacks = callbacks
        self.params = {}
        self.playground = None
    
    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)
    
    def set_playground(self, playground):
        self.playground = playground
        for callback in self.callbacks:
            callback.set_playground(playground)
        
    def _call_key_hook(self, key, hook, value=None, logs=None):
        """ Helper func for {step|episode|cycle|run}_{begin|end} methods. """

        if len(self.callbacks) == 0:
            return
        
        hook_name = f'on_{key}_{hook}'
        t_begin_name = f't_{key}_begin'
        dt_name = f'dt_{key}'

        if hook == 'begin':
            setattr(self, t_begin_name, time.time())
        
        if hook == 'end':
            t_begin = getattr(self, t_begin_name)
            dt = time.time() - t_begin
            setattr(self, dt_name, dt)
            if logs is not None:
                logs.update({dt_name: dt})
        
        for callback in self.callbacks:
            step_hook = getattr(callback, hook_name)
            if logs is not None:
                step_hook(logs) if value is None else step_hook(value, logs)
    
    def on_step_begin(self, step, logs=None):
        self._call_key_hook('step', 'begin', step , logs)

    def on_step_end(self, step, logs=None):
        self._call_key_hook('step', 'end', step , logs)
    
    def on_episode_begin(self, episode, logs=None):
        self._call_key_hook('episode', 'begin', episode , logs)

    def on_episode_end(self, episode, logs=None):
        self._call_key_hook('episode', 'end', episode , logs)

    def on_cycle_begin(self, episode, logs=None):
        self._call_key_hook('cycle', 'begin', episode , logs)

    def on_cycle_end(self, episode, logs=None):
        self._call_key_hook('cycle', 'end', episode , logs)

    def on_run_begin(self, logs=None):
        self._call_key_hook('run', 'begin', logs=logs)

    def on_run_end(self, logs=None):
        self._call_key_hook('run', 'end', logs=logs)


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

    def __init__(self, detailed_step_only_metrics=['observation', 'action', 'next_observation'],
                       step_only_metrics=['done'],
                       step_metrics=['reward', 'loss', 'exploration~exp', 'learning_rate~lr', 'dt_step~'],
                       episode_only_metrics=['dt_episode~'], 
                       episode_metrics=['reward.sum', 'loss', 'exploration~exp.last', 'learning_rate~lr.last', 'dt_step~'],
                       cycle_only_metrics=['dt_episode~'],
                       cycle_metrics=['reward', 'loss', 'exploration~exp.last', 'learning_rate~lr.last', 'dt_step~'],
                 ):
        self.cycle_only_metrics = MetricList(cycle_only_metrics)
        self.cycle_metrics = MetricList(cycle_metrics)

        self.episode_only_metrics = MetricList(episode_only_metrics)
        self.episode_metrics = MetricList(episode_metrics)

        self.step_only_metrics = MetricList(step_only_metrics)
        self.step_metrics = MetricList(step_metrics)

        self.detailed_step_only_metrics = MetricList(detailed_step_only_metrics)

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
            seen = getattr(self, metric_seen)
            new_seen = seen + 1
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
        if metric_name in logs:
            _logs = logs
        else:
            if agent_id is not None:
                agent_logs = logs.get(f'agent_{agent_id}')
                _logs = _search_logs(metric_name, agent_logs)
            else:
                _logs = _search_logs(metric_name, _logs)

        value = _logs.get(metric_name) if _logs is not None else 'N/A'
        return value

    def _update_metrics(self, metric_list:MetricList, target_prefix, source_prefix=None, logs=None, agent_id=None, reset=False):
        """ Update the logger attributes based on a metric list """
        for metric in metric_list:
            target_name = self._get_attr_name(target_prefix, metric, agent_id)

            if reset:
                self._reset_attr(target_name, metric.operator)
                continue

            # Search for source-wise attr
            if source_prefix is not None:
                src_name = self._get_attr_name(source_prefix, metric, agent_id)
                src_value = getattr(self, src_name, 'N/A')
            else:
                src_value = 'N/A'

            # If not found or no source, search in logs directly
            if src_value == 'N/A':
                src_value = self._extract_metric_from_logs(metric.name, logs, agent_id)

            # Update attr if a value was found
            if src_value != 'N/A':
                self._update_attr(target_name, src_value, metric.operator)
    
    def _update_metrics_all_agents(self, metric_list:MetricList, target_prefix, **kwargs):
        if self.n_agents > 1:
            for agent_id in range(self.n_agents):
                self._update_metrics(metric_list, target_prefix, agent_id=agent_id, **kwargs)
        else:
            self._update_metrics(metric_list, target_prefix, **kwargs)

    def on_step_end(self, step, logs={}):
        agent_id = logs.get('agent_id') if self.n_agents > 1 else None
        self._update_metrics(self.episode_metrics, 'episode', logs=logs, agent_id=agent_id)

    def on_episode_begin(self, episode, logs=None):
        self._update_metrics_all_agents(self.episode_metrics, 'episode', logs=logs, reset=True)

    def on_episode_end(self, episode, logs=None):
        if self.params['verbose'] == 1:
            self._update_metrics_all_agents(self.cycle_metrics, 'cycle', source_prefix='episode', logs=logs)

    def on_cycle_begin(self, episode, logs=None):
        if self.params['verbose'] == 1:
            self._update_metrics_all_agents(self.cycle_metrics, 'cycle', source_prefix='episode', logs=logs, reset=True)

    def on_run_begin(self, logs=None):
        self.n_agents = len(self.playground.agents)

class Logger(LoggingCallback):

    """ Default logger in every :class:`~learnrl.core.Playground` run
    
    This will print relevant informations in console.

    You can regulate the flow of informations with the argument `verbose` in :meth:`~learnrl.playground.Playground.run` directly :
     - 0 is silent (nothing will be printed)
     - 1 is only cycles of episodes (aggregated metrics over multiple episodes)
     - 2 is every episode (aggregated metrics over all steps)
     - 3 is every step (scalar metrics of all steps)
     - 4 is every step detailed (all metrics of all steps)
    
    You can also replace it with you own :class:`~learnrl.callbacks.Logger` with the argument `logger` in :meth:`~learnrl.playground.Playground.run`.
     - To build you own logger, you have to chose what metrics will be displayed and how will metrics be aggregated over steps/episodes/cycles.
       To do that, see the :ref:`metric_code` format.
    
    Parameters
    ----------
        detailed_step_only_metrics: list(str)
            Metrics to display only on detailed steps.
        step_only_metrics: list(str)
            Metrics to display only on steps.
        step_metrics: list(str)
            Metrics to display on steps and to aggregate in episodes
        episode_only_metrics: list(str)
            Metrics to display only on episodes.
        episode_metrics: list(str)
            Metrics to display on episodes and to aggregate in episodes_cycles.
        cycle_only_metrics: list(str)
            Metrics to display only on cycles.
        cycle_metrics: list(str)
            Metrics to display on cycles (aggregated from episodes and/or steps).
        titles_on_top: bool
            If true, titles will be displayed on top and not at every line in the console.
    
    """

    def __init__(self, detailed_step_only_metrics=['observation', 'action', 'next_observation'],
                       step_only_metrics=['done'],
                       step_metrics=['reward', 'loss', 'exploration~exp', 'learning_rate~lr', 'dt_step~'],
                       episode_only_metrics=['dt_episode~'], 
                       episode_metrics=['reward.sum', 'loss', 'exploration~exp.last', 'learning_rate~lr.last', 'dt_step~'],
                       cycle_metrics=['reward', 'loss', 'exploration~exp.last', 'learning_rate~lr.last', 'dt_step~'],
                       cycle_only_metrics=['dt_episode~'],
                       titles_on_top=True
                 ):
        super().__init__(detailed_step_only_metrics=detailed_step_only_metrics,
                         step_only_metrics=step_only_metrics,
                         step_metrics=step_metrics,
                         episode_only_metrics=episode_only_metrics,
                         episode_metrics=episode_metrics,
                         cycle_metrics=cycle_metrics,
                         cycle_only_metrics=cycle_only_metrics)

        self._bar_lenght = 100
        self._number_window = 9
        self.titles_on_top = titles_on_top

    def on_step_begin(self, step, logs=None):
        text = f'Step {step+1}'
        if self.params['verbose'] == 3:
            print(text + ' ' * (4 - len(str(step+1))), end=' | ')
        elif self.params['verbose'] > 3:
            self._print_bar('-', text)
    
    def on_step_end(self, step, logs={}):
        super().on_step_end(step, logs=logs)

        agent_id = logs.get('agent_id')
        verbose = self.params['verbose']
        titles_on_top = False if verbose == 4 else self.titles_on_top

        if verbose > 2:
            sep, end = (' | ', '\n') if verbose == 3 else (None, '')
            if self.n_agents > 1:
                print(f"Agent {agent_id}", end=sep)
            self._print_metrics(self.step_only_metrics + self.step_metrics, 'logs', agent_id=agent_id, logs=logs, sep=sep, end=end, titles_on_top=titles_on_top)                

        if verbose == 4:
            self._print_metrics(self.detailed_step_only_metrics, 'logs', logs=logs, titles_on_top=titles_on_top)
            self._print_bar('-')

    def on_episode_begin(self, episode, logs=None):
        super().on_episode_begin(episode, logs=logs)

        verbose = self.params['verbose']
        if verbose >= 2:
            text = "Episode " + self._get_episode_text(episode)
            if verbose == 2:
                print(text, end=' | ')
            else:
                self._print_bar('=', text)
                if verbose == 3:
                    self._print_titles(self.step_only_metrics + self.step_metrics, offset=' '*20 + '|', end='\n')

    def on_episode_end(self, episode, logs=None):
        super().on_episode_end(episode, logs=logs)

        verbose = self.params['verbose']
        if verbose >= 3:
            print()
            print("Episode " + self._get_episode_text(episode), end=' | ')                
        
        if verbose >= 2:
            self._print_metrics(self.episode_only_metrics, 'logs', logs=logs, sep=' | ', titles_on_top=False)
            if self.titles_on_top:
                self._print_titles(self.episode_metrics, prefix='\n', offset=' '*12 + '|')
            for agent_id in range(self.n_agents):
                if self.n_agents > 1:
                    print(end=f'\n    Agent {agent_id} | ')    
                self._print_metrics(self.episode_metrics, 'attrs', prefix='episode', agent_id=agent_id, sep=' | ')

        if verbose > 1:
            print()
        if verbose > 2:
            self._print_bar('=')

    def on_cycle_end(self, episode, logs=None):
         if self.params['verbose'] == 1:
            print("Episode " + self._get_episode_text(episode), end=' | ')
            
            self._print_metrics(self.cycle_only_metrics, 'logs', logs=logs, sep=' | ', titles_on_top=False)
            if self.titles_on_top:
                self._print_titles(self.cycle_metrics, prefix='\n', offset=' '*12 + '|')       
            for agent_id in range(self.n_agents):
                if self.n_agents > 1: print(end=f'\n    Agent {agent_id} | ')
                self._print_metrics(self.cycle_metrics, 'attrs', prefix='cycle', agent_id=agent_id, sep=' | ')
                    
            print()

    def on_run_begin(self, logs=None):
        super().on_run_begin(logs=logs)
        self.n_digits_episodes = int(np.log10(self.params['episodes'])) + 1

    def on_run_end(self, logs=None):
        pass

    def _print_metrics(self, metric_list:MetricList, source:str, prefix=None, agent_id=None, logs=None, sep=None, end='', titles_on_top=None):
        """ Print a metric list """
        titles_on_top = titles_on_top if titles_on_top is not None else self.titles_on_top
        for metric in metric_list:
            if source.startswith('attr'):
                name = self._get_attr_name(prefix, metric, agent_id)
                value = getattr(self, name, 'N/A')
            elif source.startswith('log'):
                value = self._extract_metric_from_logs(metric.name, logs, agent_id)
            
            pass_metric = isinstance(value, str) and value == 'N/A'
            if not pass_metric:
                self._print_metric(metric, value, titles_on_top, end=sep)
        
        print(end=end)

    def _print_metric(self, metric:Metric, metric_value, titles_on_top, **kwargs):
        """ Print a single metric based on the input type """
        if metric.name.startswith('dt_'):
            level = metric.name.split('_')[1]
            metric_display = self._get_time_text(metric_value, level)
        
        elif isinstance(metric_value, (int, float, np.float32, np.float64)):
            if metric_value == 0 or (abs(metric_value) < 1000 and abs(metric_value) > 0.001):
                metric_display = f"{metric_value:.3g}"
            else:
                metric_display = f"{metric_value:.2E}"
        
        elif isinstance(metric_value, np.ndarray):
            metric_display = '\n' + str(metric_value)
        
        else:
            metric_display = str(metric_value)
        
        space = ' ' if len(metric.surname) > 0 else ''
        if titles_on_top:
            prefix = ''
        else:
            prefix = f"{metric.surname.capitalize()}" + space
        
        metric_display = metric_display + (self._number_window - len(metric_display)) * ' '
        print(prefix + metric_display, **kwargs)
    
    def _print_titles(self, metric_list:MetricList, prefix='', offset='', end=''):
        """ Print the titles of the metric list """
        print(prefix, end=offset)
        for metric in metric_list:
            if not metric.name.startswith('dt'):
                surname = metric.surname.capitalize()[:self._number_window]
                display_name = self._text_in_middle(' ', surname, self._number_window+2)
                print(display_name, end='|')
        print(end=end)

    def _print_bar(self, line, text=None, **kwargs):
        """ Print a bar of line with centered text """
        if text:
            print(self._text_in_middle(line, text, self._bar_lenght), **kwargs)
        else:
            print(line * self._bar_lenght)

    @staticmethod
    def _text_in_middle(line, text, lenght):
        semibar_lenght = (lenght - len(text)) // 2 - 1
        odd = (lenght - len(text)) % 2 == 1
        return line * semibar_lenght + f' {text} ' + line * (semibar_lenght + 1*odd)
    
    def _get_episode_text(self, episode):
        """ Get the display text for an episode """
        text = f"{episode+1}"
        text = " "*(self.n_digits_episodes - len(text)) + text
        text += f"/{self.params['episodes']}"
        return text
    
    def _get_time_text(self, dt, unit):
        """ Get the display text for a time mesurment """
        if unit == 'episode':
            unit = 'eps'
        if dt < 1e-9:
            return 'N/A         '
        elif dt < 1e-6:
            time_display = f'{dt/1e-9:.01f}'
            time_unit = 'ns'
        elif dt < 1e-3:
            time_display = f'{dt/1e-6:.01f}'
            time_unit = 'us'
        elif dt < 1:
            time_display = f'{dt/1e-3:.01f}'
            time_unit = 'ms'
        else:
            time_display = f'{dt:.01f}'
            time_unit = ' s'
        
        margin = (5 - len(time_display)) * ' '
        return margin + f'{time_display}{time_unit}/{unit}'

import importlib
tensorflow_spec = importlib.util.find_spec('tensorflow')

if tensorflow_spec is not None:
    from learnrl.callbacks.tensorboard import TensorboardCallback
else:
    class TensorboardCallback():
        def __init__(self):
            raise ImportError('Missing dependency : tensorflow >= 2.0.0')

