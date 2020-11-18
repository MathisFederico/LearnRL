# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import wandb

from learnrl.callbacks.logging_callback import LoggingCallback, MetricList


class WandbCallback(LoggingCallback):
    
    def __init__(self,
                 step_metrics=['reward', 'loss', 'exploration~exp', 'learning_rate~lr'],
                 episode_metrics=['reward.sum', 'loss', 'exploration~exp.last', 'learning_rate~lr.last'],
                 cycle_metrics=['reward', 'loss', 'exploration~exp.last', 'learning_rate~lr.last'],
                 ):
        
        super().__init__(
            step_metrics=step_metrics,
            episode_metrics=episode_metrics,
            cycle_metrics=cycle_metrics
        )

        self.episode = 0
        
    def on_step_end(self, step, logs={}):
        super().on_step_end(step, logs=logs)
        self._update_wandb('step', self.step_metrics, logs)

    def on_episode_end(self, episode, logs=None):
        super().on_episode_end(episode, logs=logs)
        self.episode = episode
        self._update_wandb('episode', self.episode_metrics)

    def on_cycle_end(self, episode, logs=None):
        super().on_cycle_end(episode, logs=logs)
        if self.params['verbose'] == 1:
            self._update_wandb('cycle', self.cycle_metrics)

    def _update_wandb(self, prefix, metrics_list:MetricList, logs=None):
        for agent_id in range(self.n_agents):
            for metric in metrics_list:
                name = self._get_attr_name(prefix, metric, agent_id)
                value = self._get_value(metric, prefix, agent_id, logs)
                if value != 'N/A': wandb.log({name: value}, commit=False)
        wandb.log({ 'episode': self.episode })

