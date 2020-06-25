import time
import sys
import numpy as np

class Callback():

    """ An object to call functions while the :class:`~learnrl.core.Playground` is running.
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
        
    def _call_key_hook(self, key, hook, value=None, logs={}):
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
            logs.update({dt_name: dt})
        
        for callback in self.callbacks:
            step_hook = getattr(callback, hook_name)
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


class Logger(Callback):

    """ Default logger in every :class:`~learnrl.core.Playground` run """

    def __init__(self, detailed_step_only_metrics=['observation', 'action', 'next_observation'],
                       step_only_metrics=['done'],
                       step_metrics=['reward', 'loss', 'exploration~exp', 'learning_rate~lr', 'dt_step~'],
                       episode_only_metrics=[], 
                       episode_metrics=['reward.sum', 'loss', 'exploration~exp.last', 'learning_rate~lr.last', 'dt_episode~', 'dt_step~'],
                       cycle_metrics=['reward~rwd', 'loss', 'exploration~exp.last', 'learning_rate~lr.last', 'dt_episode~', 'dt_step~'],
                       cycle_only_metrics=[]):

        self.cycle_only_metrics = MetricList(cycle_only_metrics)
        self.cycle_metrics = MetricList(cycle_metrics)

        self.episode_only_metrics = MetricList(episode_only_metrics)
        self.episode_metrics = MetricList(episode_metrics)

        self.step_only_metrics = MetricList(step_only_metrics)
        self.step_metrics = MetricList(step_metrics)

        self.detailed_step_only_metrics = MetricList(detailed_step_only_metrics)

        self._bar_lenght = 50    

    def on_step_begin(self, step, logs=None):
        text = f'Step {step+1}'
        if self.params['verbose'] == 3:
            print(text, end='\t| ')
        elif self.params['verbose'] > 3:
            self._print_bar(text, '-')

    def on_step_end(self, step, logs={}):
        agent_id = logs.get('agent_id')
        verbose = self.params['verbose']

        if self.n_agents > 1 and verbose > 2:
            print(f"Agent :{logs.get('agent_id')}", end='\t| ')

        if verbose > 2:
            for metric in self.step_only_metrics + self.step_metrics:
                metric_value = self._extract_metric_from_logs(metric.name, logs, agent_id)
                if metric_value != 'N/A':
                    end = '\t| ' if verbose == 3 else None
                    self._print_metric(metric, metric_value, end=end)                    
            if verbose == 3: print()

        if verbose > 3:
            for metric in self.detailed_step_only_metrics:
                metric_value = self._extract_metric_from_logs(metric.name, logs, agent_id)
                if metric_value != 'N/A':
                    self._print_metric(metric, metric_value)
            print('-'*self._bar_lenght, end='\n\n')
        
        if verbose in (1, 2):
            for metric in self.episode_metrics:
                if not metric == 'dt_episode':
                    metric_value = self._extract_metric_from_logs(metric.name, logs, agent_id)
                    if metric_value != 'N/A':
                        attr_name = 'episode_' + metric.name
                        self._update_attr(attr_name, metric_value, metric.operator)


    def on_episode_begin(self, episode, logs=None):
        for metric in self.episode_metrics:
            attrname = 'episode_' + metric.name
            setattr(self, attrname, 0)

        if self.params['verbose'] >= 2:
            text = "Episode " + self._get_episode_text(episode)
            if self.params['verbose'] == 2:
                print(text, end=' | ')
            else:
                self._print_bar(text, '=')

    def on_episode_end(self, episode, logs=None):
        verbose = self.params['verbose']
        if verbose >= 2:
            if verbose == 3:
                print()

            for metric in self.episode_only_metrics:
                metric_value = self._extract_metric_from_logs(metric, logs)
                if metric_value != 'N/A':
                    self._print_metric(metric, metric_value, end='\t| ')
            
            for metric in self.episode_metrics:
                episode_name = 'episode_' + metric.name
                episode_value = getattr(self, episode_name, 'N/A') if not metric == 'dt_episode' else logs.get(metric.name)
                if episode_value != 'N/A':
                     self._print_metric(metric, episode_value,  end='\t| ')

            print()

        if verbose > 2:
            print("="*self._bar_lenght, end='\n\n')
        
        if verbose == 1:
            for metric in self.cycle_metrics:
                episode_name = 'episode_' + metric.name
                episode_value = getattr(self, episode_name, 'N/A') if not metric == 'dt_episode' else logs.get(metric.name)
                
                cycle_name = 'cycle_' + metric.name
                self._update_attr(cycle_name, episode_value, metric.operator)

    def on_cycle_begin(self, episode, logs=None):
        self.avg_returns = np.zeros(self.n_agents)

        self.cycle_seen_episodes = 0
        self.cycle_episode_time = 0

        self.cycle_seen_steps = 0
        self.cycle_step_time = 0

        self.cycle_loss = 0

    def on_cycle_end(self, episode, logs=None):
         if self.params['verbose'] == 1:
            print("Episode " + self._get_episode_text(episode), end=' | ')
            
            for metric in self.cycle_only_metrics:
                metric_value = self._extract_metric_from_logs(metric, logs)
                if metric_value != 'N/A':
                    self._print_metric(metric, metric_value, end='\t| ')
            
            for metric in self.cycle_metrics:
                cycle_name = 'cycle_' + metric.name
                cycle_value = getattr(self, cycle_name, 'N/A')
                if cycle_value != 'N/A':
                     self._print_metric(metric, cycle_value,  end='\t| ')
            
            print()

    def on_run_begin(self, logs=None):
        self.n_agents = len(self.playground.agents)
        self.n_digits_episodes = int(np.log10(self.params['episodes'])) + 1
        if self.params['verbose'] >= 1:
            print('***** Run started *****')

    def on_run_end(self, logs=None):
        pass

    def _print_metric(self, metric, metric_value, **kwargs):
        if metric.name.startswith('dt_'):
            level = metric.name.split('_')[1]
            metric_display = self._get_time_text(metric_value, level)
        elif isinstance(metric_value, (float, np.float32, np.float64)):
            if abs(metric_value) < 100 and abs(metric_value) > 0.01:
                metric_display = f"{metric_value:.2f}"
            else:
                metric_display = f"{metric_value:.2E}"
        else:
            metric_display = str(metric_value)
        space = ' ' if len(metric.surname) > 0 else ''
        print(f"{metric.surname.capitalize()}" + space + metric_display, **kwargs)
    
    def _print_bar(self, text, line, **kwargs):
        semibar_lenght = (self._bar_lenght - len(text)) // 2 - 1
        odd = (self._bar_lenght - len(text)) % 2 == 1
        print(line * semibar_lenght + f' {text} ' + line * (semibar_lenght + 1*odd), **kwargs)

    def _update_attr(self, attr_name, last_value, operator):
        previous_value = getattr(self, attr_name, 0)

        if operator == 'avg':
            metric_seen = attr_name + '_seen'
            seen = getattr(self, metric_seen, 1)
            setattr(self, attr_name, last_value + (last_value - previous_value) / seen)
            setattr(self, metric_seen, seen + 1)
        elif operator == 'sum':
            setattr(self, attr_name, last_value + previous_value)
        elif operator == 'last':
            setattr(self, attr_name, last_value)
        else:
            raise ValueError(f'Unknowed operator {operator}')

    def _get_episode_text(self, episode):
        text = f"{episode+1}"
        text = " "*(self.n_digits_episodes - len(text)) + text
        text += f"/{self.params['episodes']}"
        return text
    
    def _get_time_text(self, dt, unit):
        unit = 'eps' if unit == 'episode' else unit
        if dt < 1e-9:
            return f'Instant'
        if dt < 1e-6:
            return f'{dt/1e-9:.01f}ns/{unit}'
        if dt < 1e-3:
            return f'{dt/1e-6:.01f}us/{unit}'
        if dt < 1:
            return f'{dt/1e-3:.01f}ms/{unit}'
        return f'{dt:.01f}s/{unit}'

    @staticmethod
    def _extract_metric_from_logs(metric_name, logs, agent_id=-1):

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
        if agent_id != -1:
            agent_logs = logs.get(f'agent_{agent_id}')
            if metric_name in logs:
                _logs = logs
            elif agent_logs is not None:
                _logs = _search_logs(metric_name, agent_logs)
        else:
            _logs = _search_logs(metric_name, _logs)
        
        value = _logs.get(metric_name) if _logs is not None else 'N/A'
        
        return value

