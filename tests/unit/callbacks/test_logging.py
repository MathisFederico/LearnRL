# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest, sys
import numpy as np

from learnrl.callbacks import LoggingCallback, Callback, CallbackList


def test_extract_from_logs():

    extract_from_logs = LoggingCallback._extract_metric_from_logs

    logs = {
        'value': 0,
        'step': 2,
        'agent_0': {
            'value': 1,
        },
        'agent_1': {
            'value': 2,
            'specific_value': 42,
        }
    }

    # We should find values
    value = extract_from_logs('value', logs)
    assert value == 0

    # Nothing should return N/A
    nothing = extract_from_logs('nothing', logs)
    assert nothing == 'N/A'

    # If no agent is specified, it should find any specific_value in agent
    specific_value = extract_from_logs('specific_value', logs)
    assert specific_value == 42

    # We should find specific agent values
    value_0 = extract_from_logs('value', logs, agent_id=0)
    assert value_0 == 1
    value_1 = extract_from_logs('value', logs, agent_id=1)
    assert value_1 == 2

    # When agent is specified, it should return outer value
    # if a specific value is not found
    step = extract_from_logs('step', logs, agent_id=0)
    assert step == 2



class DummyPlayground():

    def __init__(self, agents=[0]):
        self.agents = agents

    def run(self, callbacks, eps_end_func=None, verbose=0):

        n_episodes = 10

        callbacks = CallbackList(callbacks)
        callbacks.set_params({'verbose':verbose, 'episodes': n_episodes})
        callbacks.set_playground(self)

        logs = {}
        callbacks.on_run_begin(logs)
        
        cycle_len = 3

        for episode in range(n_episodes):

            agent_id = 0
            logs.update({'agent_id': agent_id})

            if episode % cycle_len == 0:

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

                callbacks.on_cycle_begin(episode, logs)

            callbacks.on_episode_begin(episode, logs)

            reward_sum = loss_sum = 0
            n_valid_loss = n_valid_reward = 0
            n_steps = 10

            for step in range(n_steps):

                if step % 3 == 0:
                    # callbacks.on_step_cycle_begin(step, logs)
                    pass
                
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

                callbacks.on_step_end(step, logs)

                if step % 3 == 2 or step == n_steps - 1:
                    # callbacks.on_step_cycle_end(step, logs)
                    pass
            
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

            if (episode + 1) % cycle_len == 0 or episode == n_episodes - 1:

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

                callbacks.on_cycle_end(episode, logs)
            
            if eps_end_func is not None: eps_end_func(callbacks, logs)
            
        callbacks.on_run_end(logs)


def test_logging_avg_sum():

    for eps_operator in ['avg', 'sum']:

        for cycle_operator in ['avg', 'sum']:

            print(eps_operator, cycle_operator, '\n')

            logging_callback = LoggingCallback(
                detailed_step_only_metrics=[],
                step_metrics=['reward', 'loss'],
                episode_only_metrics=[], 
                episode_metrics=[f'reward.{eps_operator}', f'loss.{eps_operator}'],
                cycle_only_metrics=[],
                cycle_metrics=[f'reward.{cycle_operator}', f'loss.{cycle_operator}'],
            )

            def check_function(callbacks, logs):
                callback_dict = callbacks.callbacks[0].__dict__

                for position in ('episode', 'cycle'):
                    for metric_name in ('reward', 'loss'):

                        if position == 'episode':
                            expected = logs.get(f'{metric_name}_{position}_{eps_operator}')
                        elif position == 'cycle':
                            expected = logs.get(f'{metric_name}_{eps_operator}_{position}_{cycle_operator}')

                        logged = callback_dict[f'{position}_{metric_name}']
                        if expected is not None:
                            print(position.capitalize(), metric_name, logged, expected)
                            assert logged != 'N/A'
                            assert np.isclose(logged, expected)
                        
            pg = DummyPlayground()
            pg.run([logging_callback], eps_end_func=check_function, verbose=1)
            print()


def test_display():

    from learnrl.callbacks import Logger

    for titles_on_top in (False, True):
        for verbose in range(4):

            logging_callback = Logger(
                titles_on_top=titles_on_top
            )

            print(f'Verbose {verbose}, Title_on_top {titles_on_top}\n')

            pg = DummyPlayground()
            pg.run([logging_callback], verbose=verbose)

            print()
            
    assert True
