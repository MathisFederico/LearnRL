# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest

from learnrl.callbacks import LoggingCallback, Callback, CallbackList

class DummyPlayground():

    def __init__(self, agents=[0]):
        self.agents = agents

    def run(self, callbacks, eps_end_func=None, verbose=0):

        callbacks = CallbackList(callbacks)
        callbacks.set_params({'verbose':verbose, 'param':'foo'})
        callbacks.set_playground(self)

        logs = {}
        callbacks.on_run_begin(logs)
        n_episodes = 10
        cycle_len = 3

        for episode in range(n_episodes):

            agent_id = 0
            logs.update({'agent_id': agent_id})

            if episode % cycle_len == 0:
                value_sum_cycle_sum = 0
                value_avg_cycle_sum = 0
                cycle_episode_seen = 0
                callbacks.on_cycle_begin(episode, logs)

            callbacks.on_episode_begin(episode, logs)
            value_sum = 0
            n_steps = 10

            for step in range(n_steps):

                if step % 3 == 0:
                    # callbacks.on_step_cycle_begin(step, logs)
                    pass

                value = step + 10 * episode
                value_sum += value
                logs.update({'value': value})
                callbacks.on_step_begin(step, logs)

                ## Step ##
                callbacks.on_step_end(step, logs)
            
                if step % 3 == 2 or step == n_steps - 1:
                    # callbacks.on_step_cycle_end(step, logs)
                    pass
            
            value_avg = value_sum / n_steps
            logs.update({'value_episode_avg': value_avg, 'value_episode_sum': value_sum})

            ## Done ##
            callbacks.on_episode_end(episode, logs)       

            value_sum_cycle_sum += value_sum
            value_avg_cycle_sum += value_avg
            cycle_episode_seen += 1

            value_sum_cycle_avg = value_sum_cycle_sum / cycle_episode_seen
            value_avg_cycle_avg = value_avg_cycle_sum / cycle_episode_seen
            logs.update({
                'value_sum_cycle_sum': value_sum_cycle_sum,
                'value_avg_cycle_sum': value_avg_cycle_sum,
                'value_sum_cycle_avg': value_sum_cycle_avg,
                'value_avg_cycle_avg': value_avg_cycle_avg
            })

            if episode % cycle_len == cycle_len - 1 or episode == n_episodes - 1:
                callbacks.on_cycle_end(episode, logs)
            
            if eps_end_func is not None: eps_end_func(callbacks, logs)
            
        callbacks.on_run_end(logs)


def test_logging_multiagent():
    pass

def test_logging_avg_sum():

    for eps_operator in ['avg', 'sum']:
        for cycle_operator in ['avg', 'sum']:
            logging_callback = LoggingCallback(
                step_metrics=['value'],
                episode_metrics=[f'value.{eps_operator}'],
                cycle_metrics=[f'value.{cycle_operator}']
            )

            def check_function(callbacks, logs):
                callback_dict = callbacks.callbacks[0].__dict__
                assert callback_dict['episode_value'] == logs.get(f'value_episode_{eps_operator}')
                assert callback_dict['cycle_value'] == logs.get(f'value_{eps_operator}_cycle_{cycle_operator}')

            pg = DummyPlayground()
            pg.run([logging_callback], eps_end_func=check_function, verbose=1)

