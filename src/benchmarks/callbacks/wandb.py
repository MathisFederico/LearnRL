"""WandbCallback to log metrics into weight&biases

https://wandb.ai/site

"""

from typing import List, Union

import wandb
from benchmarks.callbacks.logging_callback import LoggingCallback, MetricList


def construct_panel_configs(metric_name: str, title: str = None):
    """Construct a custom wandb query to build the custom graphs."""

    histories = [
        {
            "name": "history",
            "args": [
                {
                    "name": "keys",
                    "value": [
                        "step",
                        "steps_cycle",
                        "episode",
                        "episodes_cycle",
                        f"step_{metric_name}",
                        f"steps_cycle_{metric_name}",
                        f"episode_{metric_name}",
                        f"episodes_cycle_{metric_name}",
                    ],
                }
            ],
            "fields": [],
        }
    ]

    user_query = {
        "queryFields": [
            {
                "name": "runSets",
                "args": [{"name": "runSets", "value": "${runSets}"}],
                "fields": [
                    {"name": "id", "fields": []},
                    {"name": "name", "fields": []},
                    {"name": "_defaultColorIndex", "fields": []},
                ]
                + histories,
            }
        ],
    }

    title = title or metric_name.capitalize()
    fields = {
        "title": title,
        "x1": "step",
        "x2": "steps_cycle",
        "x3": "episode",
        "x4": "episodes_cycle",
        "y1": f"step_{metric_name}",
        "y2": f"steps_cycle_{metric_name}",
        "y3": f"episode_{metric_name}",
        "y4": f"episodes_cycle_{metric_name}",
    }

    return {
        "userQuery": user_query,
        "panelDefId": "mathisfederico/benchmarks-chart",
        "transform": {"name": "tableWithLeafColNames"},
        "fieldSettings": fields,
    }


class WandbCallback(LoggingCallback):

    """WandbCallback will log metrics to wandb."""

    def __init__(
        self,
        run,
        metrics: List[Union[str, tuple]] = None,
        detailed_step_metrics: List[str] = None,
        episode_only_metrics: List[str] = None,
    ):
        """WandbCallback will log metrics to wandb.

        See https://wandb.ai.

        Args:
            metrics: list(str) or list(tuple)
                Metrics to display and how to aggregate them.
            detailed_step_metrics: list(str)
                Metrics to display only on detailed steps.
            episode_only_metrics: list(str)
                Metrics to display only on episodes.
        """

        super().__init__(
            metrics=metrics,
            detailed_step_metrics=detailed_step_metrics,
            episode_only_metrics=episode_only_metrics,
        )

        self.metrics = metrics

        self.run = run
        self.chart_is_init = False

        self.step = 0
        self.steps_cycle = 0
        self.episode = 0
        self.episodes_cycle = 0

    def on_step_begin(self, step, logs=None):
        super().on_step_begin(step, logs=logs)

    def on_step_end(self, step, logs=None):
        super().on_step_end(step, logs=logs)

        if not self.chart_is_init:
            self._init_charts()
            self.chart_is_init = True

        self._update_wandb("step", self.step_metrics, logs)
        self._update_wandb("steps_cycle", self.steps_cycle_metrics, logs)
        self._update_wandb("episode", self.episode_metrics, logs)
        self._update_wandb("episode", self.episode_only_metrics, logs)
        self._update_wandb("episodes_cycle", self.episodes_cycle_metrics, logs)
        wandb.log(
            {
                "step": self.step,
                "episode": self.episode,
                "steps_cycle": self.steps_cycle,
                "episodes_cycle": self.episodes_cycle,
            }
        )
        self.step += 1

    def on_steps_cycle_end(self, step, logs=None):
        super().on_steps_cycle_end(step, logs=logs)
        self.steps_cycle += 1

    def on_episode_end(self, episode, logs=None):
        super().on_episode_end(episode, logs=logs)
        self.episode += 1

    def on_episodes_cycle_end(self, episode, logs=None):
        super().on_episodes_cycle_end(episode, logs=logs)
        self.episodes_cycle += 1

    def _init_charts(self):
        metrics_with_chart = []
        allmetrics = (
            self.step_metrics
            + self.steps_cycle_metrics
            + self.episode_metrics
            + self.episodes_cycle_metrics
        )
        for metric in allmetrics:
            metric_name = metric.name
            if not metric_name in metrics_with_chart:
                title = f"{metric_name.capitalize()} ({metric.surname.capitalize()})"
                panel_config = construct_panel_configs(metric_name, title)
                panel_name = metric_name + "_panel"
                self.run._add_panel(
                    panel_name, "Vega2", panel_config
                )  # pylint: disable=protected-access
                metrics_with_chart.append(metric_name)

    def _update_wandb(self, prefix, metrics_list: MetricList, logs=None):
        for agent_id in range(self.n_agents):
            for metric in metrics_list:
                name = self._get_attr_name(prefix, metric, agent_id)
                value = self._get_value(metric, prefix, agent_id, logs)
                if value != "N/A":
                    wandb.log({name: value}, commit=False)
        wandb.log({"episode": self.episode})
