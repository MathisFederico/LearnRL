# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import wandb

from learnrl.callbacks.logging_callback import LoggingCallback, MetricList


class WandbCallback(LoggingCallback):

    """ WandbCallback logger if wandb is installed.

    Parameters
    ---------
        metrics: list(str) or list(tuple)
            Metrics to display and how to aggregate them.
        detailed_step_metrics: list(str)
            Metrics to display only on detailed steps.
        episode_only_metrics: list(str)
            Metrics to display only on episodes.
    """

    def __init__(self, metrics=(('reward', {'steps': 'sum', 'episode': 'sum'})),
                detailed_step_metrics=('observation', 'action', 'next_observation'),
                episode_only_metrics=('dt_episode~')):

        super().__init__(
            metrics=metrics,
            detailed_step_metrics=detailed_step_metrics,
            episode_only_metrics=episode_only_metrics
        )

        self.episode = 0

    def on_step_end(self, step, logs=None):
        super().on_step_end(step, logs=logs)
        self._update_wandb('step', self.step_metrics, logs)

    def on_episode_end(self, episode, logs=None):
        super().on_episode_end(episode, logs=logs)
        self.episode = episode
        self._update_wandb('episode', self.episode_metrics)

    def _update_wandb(self, prefix, metrics_list:MetricList, logs=None):
        for agent_id in range(self.n_agents):
            for metric in metrics_list:
                name = self._get_attr_name(prefix, metric, agent_id)
                value = self._get_value(metric, prefix, agent_id, logs)
                if value != 'N/A':
                    wandb.log({name: value}, commit=False)
        wandb.log({ 'episode': self.episode })
