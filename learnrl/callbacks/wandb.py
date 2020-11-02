import wandb

from learnrl.callbacks import LoggingCallback, MetricList


class WandbLogger(LoggingCallback):
    
    def __init__(self,
                 step_metrics=['reward', 'loss', 'exploration~exp', 'learning_rate~lr'],
                 episode_metrics=['reward.sum', 'loss', 'exploration~exp.last', 'learning_rate~lr.last'],
                 cycle_metrics=['reward', 'loss', 'exploration~exp.last', 'learning_rate~lr.last'],
                 ):
        super().__init__(step_metrics=step_metrics,
                    episode_metrics=episode_metrics,
                    cycle_metrics=cycle_metrics)
        self.step = 1 # Internal step counter
        
    def on_step_end(self, step, logs={}):
        super().on_step_end(step, logs=logs)
        self._update_wandb(self.step, 'step', self.step_metrics, logs)
        self.step += 1

    def on_episode_end(self, episode, logs=None):
        super().on_episode_end(episode, logs=logs)
        self._update_wandb(episode + 1, 'episode', self.episode_metrics)

    def on_cycle_end(self, episode, logs=None):
        super().on_cycle_end(episode, logs=logs)
        if self.params['verbose'] == 1:
            self._update_wandb(episode + 1, 'cycle', self.cycle_metrics)

    def _update_wandb(self, step, prefix, metrics_list:MetricList, logs=None):
        for agent_id in range(self.n_agents):
            for metric in metrics_list:
                name = self._get_attr_name(prefix, metric, agent_id)                        
                if logs is None:
                    value = getattr(self, name, 'N/A')
                else:
                    value = self._extract_metric_from_logs(metric.name, logs, agent_id)

                if value != 'N/A':
                    wandb.log({name: value})

