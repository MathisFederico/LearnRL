import numpy as np

from learnrl.callbacks import LoggingCallback, MetricList, Metric


class Logger(LoggingCallback):

    """ Default logger in every :class:`~learnrl.playground.Playground` run
    
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

    def __init__(self, 
                detailed_step_only_metrics=['observation', 'action', 'next_observation'],
                step_metrics=['reward', 'loss', 'exploration~exp', 'learning_rate~lr'],
                episode_only_metrics=[], 
                episode_metrics=['reward.sum', 'loss', 'exploration~exp.last', 'learning_rate~lr.last'],
                cycle_metrics=['reward', 'loss', 'exploration~exp.last', 'learning_rate~lr.last'],
                cycle_only_metrics=[],
                titles_on_top=True
                ):
        
        super().__init__(
            detailed_step_only_metrics=detailed_step_only_metrics,
            step_metrics=step_metrics,
            episode_only_metrics=episode_only_metrics,
            episode_metrics=episode_metrics,
            cycle_metrics=cycle_metrics,
            cycle_only_metrics=cycle_only_metrics
        )

        self._bar_lenght = 100
        self._number_window = 9
        self.titles_on_top = titles_on_top

    def on_step_begin(self, step, logs=None):
        text = self._get_step_text(step, pad=self.verbose == 3)
        if self.verbose == 3:
            print(text, end=' | ')
        elif self.verbose == 4:
            self._print_bar('-', text)
    
    def on_step_end(self, step, logs={}):
        super().on_step_end(step, logs=logs)

        agent_id = logs.get('agent_id')

        if self.verbose > 2:
            sep, end = (' | ', '\n') if self.verbose == 3 else (None, '')
            if self.n_agents > 1:
                print(f"Agent {agent_id}", end=sep)
            self._print_metrics(self.step_metrics, 'logs', agent_id=agent_id, logs=logs, sep=sep, end=end)                

        if self.verbose == 4:
            self._print_metrics(self.detailed_step_only_metrics, 'logs', logs=logs)
            self._print_bar('-')

    def on_episode_begin(self, episode, logs=None):
        super().on_episode_begin(episode, logs=logs)

        if self.verbose >= 2:
            text = "Episode " + self._get_episode_text(episode)
            if self.verbose == 2:
                print(text, end=' | ')
            else:
                self._print_bar('=', text)
                if self.verbose == 3 and self.titles_on_top:
                    step_text = self._get_step_text(0)
                    self._print_titles(self.step_metrics, offset=' ' * len(step_text) + ' |', end='\n')

    def on_episode_end(self, episode, logs=None):
        super().on_episode_end(episode, logs=logs)

        if self.verbose >= 3:
            print()
            print("Episode " + self._get_episode_text(episode), end=' | ')                
        
        if self.verbose >= 2:
            if self.titles_on_top and self.n_agents > 1:
                self._print_titles(self.episode_metrics, prefix='\n', offset=' '*12 + '|')
            
            for agent_id in range(self.n_agents):
                if self.n_agents > 1:
                    print(end=f'\n    Agent {agent_id} | ')
                
                titles_on_top = False if self.verbose >= 3 else None

                self._print_metrics(self.episode_metrics, 'attrs', prefix='episode',
                    agent_id=agent_id, sep=' | ', titles_on_top=titles_on_top
                )
            
            self._print_metrics(self.episode_only_metrics, 'logs', logs=logs, sep=' | ', titles_on_top=False)

        if self.verbose > 1:
            print()

        if self.verbose > 2:
            self._print_bar('=')

    def on_cycle_end(self, episode, logs=None):
         if self.verbose == 1:
            print("Episode " + self._get_episode_text(episode), end=' | ')
            
            self._print_metrics(self.cycle_only_metrics, 'logs', logs=logs, sep=' | ')

            if self.titles_on_top and self.n_agents > 1:
                self._print_titles(self.cycle_metrics, prefix='\n', offset=' '*12 + '|')
            
            for agent_id in range(self.n_agents):
                if self.n_agents > 1: print(end=f'\n    Agent {agent_id} | ')
                self._print_metrics(self.cycle_metrics, 'attrs', prefix='cycle', agent_id=agent_id, sep=' | ')
            
            print()

    def on_run_begin(self, logs=None):
        super().on_run_begin(logs=logs)
        self.n_digits_episodes = int(np.log10(self.params['episodes'])) + 1
        self.verbose = self.params['verbose']

        if self.verbose == 4:
            self.titles_on_top = False

        if self.titles_on_top and self.n_agents == 1 and self.verbose in (1, 2):
            offset_len = len("Episode " + self._get_episode_text(0))
            metrics_to_print = self.cycle_metrics if self.verbose == 1 else self.episode_metrics
            self._print_titles(metrics_to_print, prefix='', offset=' ' * offset_len + ' |', end='\n')

    def on_run_end(self, logs=None):
        pass

    def _print_metrics(self, metric_list:MetricList, source:str, prefix=None, agent_id=None,
                             logs=None, sep=None, end='', titles_on_top=None):
        """ Print a metric list """
        titles_on_top = titles_on_top if titles_on_top is not None else self.titles_on_top
        for metric in metric_list:
            value = self._get_value(metric, prefix, agent_id, logs)
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
        prefix = '' if titles_on_top else f"{metric.surname.capitalize()}" + space

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
        text = " " * (self.n_digits_episodes - len(text)) + text
        text += f"/{self.params['episodes']}"
        return text

    def _get_step_text(self, step, pad=True):
        text = f'Step {step+1}'
        if pad and len(text) < 11:
            text += ' ' * (11 - len(text))
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
            time_unit = 's '
        
        margin = (5 - len(time_display)) * ' '
        return margin + f'{time_display}{time_unit}/{unit}'

