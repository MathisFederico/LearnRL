import datetime
import tensorflow as tf

from learnrl.callbacks import LoggingCallback

class TensorboardCallback(LoggingCallback):

    """ Tensorboard logger if tensorflow is installed.

    Parameters
    ---------
        run_name: :class:`str`
            Specify a run name for Tensorboard, default is a datetime.
        log_dir: :class:`str`
            Logs directory, default is './logs/'.
        step_metrics: list(str)
            Metrics to log on steps and to aggregate in episodes.
        episode_metrics: list(str)
            Metrics to log on episodes and to aggregate in episodes_cycles.
        cycle_metrics: list(str)
            Metrics to log on cycles (aggregated from episodes and/or steps).
    """

    def __init__(self,
                 run_name=None,
                 log_dir='./logs/',
                 step_metrics=['reward', 'loss', 'exploration~exp', 'learning_rate~lr', 'dt_step~'],
                 episode_metrics=['reward.sum', 'loss', 'exploration~exp.last', 'learning_rate~lr.last', 'dt_step~'],
                 cycle_metrics=['reward', 'loss', 'exploration~exp.last', 'learning_rate~lr.last', 'dt_step~'],
                 ):
        super().__init__(step_metrics=step_metrics,
                         episode_metrics=episode_metrics,
                         cycle_metrics=cycle_metrics)

        filepath = log_dir
        filepath += datetime.datetime.now().strftime("%Y%m%d-%H%M%S") if run_name is None else run_name
        self.filepath = filepath
        self.writer = tf.summary.create_file_writer(self.filepath)

        self.step = 1 # Internal step counter

    def on_step_end(self, step, logs={}):
        super().on_step_end(step, logs=logs)
        self._update_tensorboard(self.step, 'step', self.step_metrics, logs)
        self.step += 1

    def on_episode_end(self, episode, logs=None):
        super().on_episode_end(episode, logs=logs)
        self._update_tensorboard(episode + 1, 'episode', self.episode_metrics)


    def on_cycle_end(self, episode, logs=None):
        super().on_cycle_end(episode, logs=logs)
        if self.params['verbose'] == 1: self._update_tensorboard(episode + 1, 'cycle', self.cycle_metrics)

    def _update_tensorboard(self, step, prefix, metrics_list, logs=None, a=None):
        """ Helper function for writing new values to Tensorboard summary.

        Parameters
        ----------
            step: :class:`int`
                Step value for the Tensorboard summary.
            prefix: :class:`str`
                Prefix for metrics name.
            metrics_list: :class:`~learnrl.callbacks.MetricList`
                Metrics to write.
            logs: :class:`dict` (default is None)
                If set to None, metrics value will be searched in attributes, otherwise they will be searched in logs.

            """
        with self.writer.as_default(): #pylint: disable=all
            for agent_id in range(self.n_agents):
                for metric in metrics_list:
                    name = self._get_attr_name(prefix, metric, agent_id)
                    value = getattr(self, name, 'N/A') if logs is None else self._extract_metric_from_logs(metric.name, logs, agent_id)

                    if value != 'N/A': tf.summary.scalar(name, value, step=step)
            self.writer.flush()
