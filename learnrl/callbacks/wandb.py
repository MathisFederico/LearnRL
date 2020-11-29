# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import wandb

from learnrl.callbacks.logging_callback import LoggingCallback, MetricList


class WandbCallback(LoggingCallback):
    
    def __init__(self,
                detailed_step_metrics=['observation', 'action', 'next_observation'],
                episode_only_metrics=['dt_episode~'],
                metrics=[
                    ('reward', {'steps': 'sum', 'episode': 'sum'}),
                ]):
        
        super().__init__(
            detailed_step_metrics=detailed_step_metrics,
            episode_only_metrics=episode_only_metrics,
            metrics=metrics
        )

        self.episode = 0
        
    def on_step_end(self, step, logs={}):
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
                if value != 'N/A': wandb.log({name: value}, commit=False)
        wandb.log({ 'episode': self.episode })

