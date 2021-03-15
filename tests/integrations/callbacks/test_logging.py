# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

""" Test metrics logging behavior """

import pytest
import numpy as np

from learnrl.callbacks.callback import CallbackList
from learnrl.callbacks.logging_callback import LoggingCallback
from learnrl.callbacks import Logger


class DummyPlayground():
    """ Dummy playground for testing """

    def __init__(self, agents=None):
        self.agents = agents if agents is not None else [0]

    def run(self, callbacks, eps_end_func=None, verbose=0):

        n_episodes = 10
        steps_cycle_len = 3
        episodes_cycle_len = 3

        callbacks = CallbackList(callbacks)
        callbacks.set_params({
            'verbose':verbose,
            'episodes': n_episodes,
            'episodes_cycle_len':episodes_cycle_len,
            'steps_cycle_len':steps_cycle_len,
        })
        callbacks.set_playground(self)

        logs = {}
        callbacks.on_run_begin(logs)

        for episode in range(n_episodes):

            agent_id = 0
            logs.update({'agent_id': agent_id})

            if episode % episodes_cycle_len == 0:

                cycle_episode_seen = 0
                cycle_tracker = {}

                for metric_name in ('reward', 'loss'):

                    cycle_tracker[metric_name] = {'sum_cycle_sum': 0, 'avg_cycle_sum': 0}

                    logs.update({
                        f'{metric_name}_sum_cycle_sum': None,
                        f'{metric_name}_avg_cycle_sum': None,
                        f'{metric_name}_sum_cycle_avg': None,
                        f'{metric_name}_avg_cycle_avg': None
                    })

                callbacks.on_episodes_cycle_begin(episode, logs)

            callbacks.on_episode_begin(episode, logs)

            reward_sum = loss_sum = 0
            n_valid_loss = n_valid_reward = 0
            n_steps = 10

            for step in range(n_steps):

                if step % steps_cycle_len == 0:

                    steps_cycle_seen = 0
                    steps_cycle_tracker = {}

                    for metric_name in ('reward', 'loss'):
                        steps_cycle_tracker[metric_name] = 0

                        logs.update({
                            f'{metric_name}_steps_sum': None,
                            f'{metric_name}_steps_avg': None
                        })

                    callbacks.on_steps_cycle_begin(step, logs)

                callbacks.on_step_begin(step, logs)

                ## Step ##

                reward = 1 + step + 10 * episode
                reward_sum += reward
                n_valid_reward += 1

                if step == 0 and episode == 0:
                    loss = 'N/A'
                else:
                    n_valid_loss += 1
                    loss = 1 / (reward + 1) ** 1.6
                    loss_sum += loss

                logs.update({'reward': reward})
                logs.update({'agent_0':{'loss':loss}})

                steps_cycle_tracker['reward'] += reward
                if not isinstance(loss, str):
                    steps_cycle_tracker['loss'] += loss
                steps_cycle_seen += 1

                callbacks.on_step_end(step, logs)

                if (step + 1) % steps_cycle_len == 0 or step == n_steps - 1:
                    for metric_name in ('reward', 'loss'):
                        logs.update({
                            f'{metric_name}_steps_sum': steps_cycle_tracker[metric_name],
                            f'{metric_name}_steps_avg': steps_cycle_tracker[metric_name] / steps_cycle_seen,
                        })
                    callbacks.on_steps_cycle_end(step, logs)

            reward_avg = reward_sum / n_valid_reward
            loss_avg = loss_sum / n_valid_loss

            for metric_name, metric_avg, metric_sum in zip(
                ('reward', 'loss'),
                (reward_avg, loss_avg),
                (reward_sum, loss_sum),
            ):

                logs.update({
                    f'{metric_name}_episode_avg': metric_avg,
                    f'{metric_name}_episode_sum': metric_sum,
                    }
                )

                cycle_tracker[metric_name]['sum_cycle_sum'] += metric_sum
                cycle_tracker[metric_name]['avg_cycle_sum'] += metric_avg

            ## Done ##
            callbacks.on_episode_end(episode, logs)
            cycle_episode_seen += 1

            if (episode + 1) % episodes_cycle_len == 0 or episode == n_episodes - 1:

                for metric_name, metric_avg, metric_sum in zip(
                        ('reward', 'loss'),
                        (reward_avg, loss_avg),
                        (reward_sum, loss_sum),
                    ):

                    sum_cycle_avg = cycle_tracker[metric_name]['sum_cycle_sum'] / cycle_episode_seen
                    avg_cycle_avg = cycle_tracker[metric_name]['avg_cycle_sum'] / cycle_episode_seen

                    logs.update({
                        f'{metric_name}_sum_cycle_sum': cycle_tracker[metric_name]['sum_cycle_sum'],
                        f'{metric_name}_avg_cycle_sum': cycle_tracker[metric_name]['avg_cycle_sum'],
                        f'{metric_name}_sum_cycle_avg': sum_cycle_avg,
                        f'{metric_name}_avg_cycle_avg': avg_cycle_avg
                    })

                callbacks.on_episodes_cycle_end(episode, logs)

            if eps_end_func is not None:
                eps_end_func(callbacks, logs)

        callbacks.on_run_end(logs)


@pytest.mark.parametrize('cycle_operator', ['avg', 'sum'])
@pytest.mark.parametrize('metric_name', ['reward', 'loss'])
def test_logging_steps_operators_(cycle_operator, metric_name):

    print(cycle_operator, metric_name)

    logging_callback = LoggingCallback(
        metrics=[('reward', {'steps':cycle_operator}), ('loss', {'steps':cycle_operator})],
    )

    def check_function(callbacks, logs):
        callback_dict = callbacks.callbacks[0].__dict__
        expected = logs.get(f'{metric_name}_steps_{cycle_operator}')
        logged = callback_dict[f'steps_cycle_{metric_name}']
        if expected is not None:
            print(metric_name, logged, expected)
            assert logged != 'N/A' and np.isclose(logged, expected), \
                f'Logged {logged} instead of {expected}'

    playground = DummyPlayground()
    playground.run([logging_callback], eps_end_func=check_function, verbose=3)
    print()

@pytest.mark.parametrize('eps_operator', ['avg', 'sum'])
@pytest.mark.parametrize('cycle_operator', ['avg', 'sum'])
@pytest.mark.parametrize('metric_name', ['reward', 'loss'])
def test_logging_episodes_operators_(eps_operator, cycle_operator, metric_name):

    print(eps_operator, cycle_operator, '\n')

    logging_callback = LoggingCallback(
        detailed_step_metrics=[],
        metrics=[('reward', {'episode': eps_operator, 'episodes': cycle_operator}),
                 ('loss', {'episode': eps_operator, 'episodes': cycle_operator})],
    )

    def check_function(callbacks, logs):
        callback_dict = callbacks.callbacks[0].__dict__

        for position in ('episode', 'episodes_cycle'):
            if position == 'episode':
                expected = logs.get(f'{metric_name}_episode_{eps_operator}')
            elif position == 'episodes_cycle':
                expected = logs.get(f'{metric_name}_{eps_operator}_cycle_{cycle_operator}')

            logged = callback_dict[f'{position}_{metric_name}']
            if expected is not None:
                print(position.capitalize(), metric_name, logged, expected)
                assert logged != 'N/A' and np.isclose(logged, expected), \
                    f'Logged {logged} instead of {expected}'

    playground = DummyPlayground()
    playground.run([logging_callback], eps_end_func=check_function, verbose=1)
    print()


def test_display():
    """ display """
    for titles_on_top in (False, True):
        for verbose in range(5):

            logging_callback = Logger(
                titles_on_top=titles_on_top
            )

            print(f'Verbose {verbose}, Title_on_top {titles_on_top}\n')

            playground = DummyPlayground()
            playground.run([logging_callback], verbose=verbose)

            print()
