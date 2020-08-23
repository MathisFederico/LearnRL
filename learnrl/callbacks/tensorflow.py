import importlib

tensorflow_spec = importlib.util.find_spec('tensorflow')

if tensorflow_spec is not None:
    import tensorflow as tf
    import datetime

    from learnrl.callbacks import LoggingCallback

    class TensorboardCallback(LoggingCallback):

        """ Tensorboard logger if tensorflow is installed. """

        def __init__(self,
                     log_dir='./logs/',
                     episode_metrics=['reward.sum', 'loss', 'exploration~exp.last', 'learning_rate~lr.last', 'dt_step~'],
                     cycle_metrics=['reward', 'loss', 'exploration~exp.last', 'learning_rate~lr.last', 'dt_step~'],
                     ):
            super().__init__(episode_metrics=episode_metrics,
                             cycle_metrics=cycle_metrics)

            self.filepath = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            self.writer = tf.summary.create_file_writer(self.filepath)


        def on_episode_end(self, episode, logs=None):
            super().on_episode_end(episode, logs=logs)

            with self.writer.as_default():
                for agent_id in range(self.n_agents):
                    for metric in self.episode_metrics:
                        name = self._get_attr_name('episode', metric, agent_id)
                        value = getattr(self, name, 'N/A')

                        if value != 'N/A': tf.summary.scalar(name, value, step=episode + 1)
                self.writer.flush()

        def on_cycle_end(self, episode, logs=None):
            super().on_cycle_end(episode, logs=logs)
            if self.params['verbose'] > 1: return

            with self.writer.as_default():
                for agent_id in range(self.n_agents):
                    for metric in self.cycle_metrics:
                        name = self._get_attr_name('cycle', metric, agent_id)
                        value = getattr(self, name, 'N/A')

                        if value != 'N/A': tf.summary.scalar(name, value, step=episode + 1)

else:
    raise ImportError('Tensorflow should be installed.')
