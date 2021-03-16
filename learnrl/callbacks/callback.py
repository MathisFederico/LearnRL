# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

"""Callback abstract classes"""

import time

class Callback():

    """ An object to call functions while the :class:`~learnrl.playground.Playground` is running.
    You can define the custom functions `on_{position}` where position can be :

    >>> run_begin
    ...     episodes_cycle_begin
    ...         episode_begin
    ...             steps_cycle_begin
    ...                 step_begin
    ...                 # env.step()
    ...                 step_end
    ...             steps_cycle_end
    ...         # done==True
    ...         episode_end
    ...     episodes_cycle_end
    ... run_end

    """

    def __init__(self):
        self.params = {}
        self.playground = None

    def set_params(self, params):
        """Sets run parameters"""
        self.params = params

    def set_playground(self, playground):
        """Sets reference to the used playground"""
        self.playground = playground

    def on_step_begin(self, step: int, logs: dict=None):
        """Triggers on each step beginning

        Args:
            step: current step.
            logs: current logs.

        """

    def on_step_end(self, step: int, logs: dict=None):
        """Triggers on each step end

        Args:
            step: current step.
            logs: current logs.

        """

    def on_steps_cycle_begin(self, step: int, logs: dict=None):
        """Triggers on each step cycle beginning

        Args:
            step: current step.
            logs: current logs.

        """

    def on_steps_cycle_end(self, step: int, logs: dict=None):
        """Triggers on each step cycle end

        Args:
            step: current step.
            logs: current logs.

        """

    def on_episode_begin(self, episode: int, logs: dict=None):
        """Triggers on each episode beginning

        Args:
            episode: current episode.
            logs: current logs.

        """

    def on_episode_end(self, episode: int, logs: dict=None):
        """Triggers on each episode end

        Args:
            episode: current episode.
            logs: current logs.

        """

    def on_episodes_cycle_begin(self, episode: int, logs: dict=None):
        """Triggers on each episode cycle beginning

        Args:
            episode: current episode.
            logs: current logs.

        """

    def on_episodes_cycle_end(self, episode: int, logs: dict=None):
        """Triggers on each episode cycle end

        Args:
            episode: current episode.
            logs: current logs.

        """

    def on_run_begin(self, logs: dict=None):
        """Triggers on each run beginning

        Args:
            logs: current logs.

        """

    def on_run_end(self, logs: dict=None):
        """Triggers on run end

        Args:
            logs: current logs.

        """


class CallbackList(Callback):
    """ An wrapper to use a list of :class:`Callback`.

    Call all concerned callbacks While the :class:`~learnrl.playground.Playground` is running.

    """

    def __init__(self, callbacks=()):
        super().__init__()
        self.callbacks = callbacks

    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)

    def set_playground(self, playground):
        self.playground = playground
        for callback in self.callbacks:
            callback.set_playground(playground)

    def _call_key_hook(self, key, hook, value=None, logs=None):
        """ Helper func for {step|episode|steps_cycle|episodes_cycle|run}_{begin|end} methods. """

        if len(self.callbacks) == 0:
            return

        hook_name = f'on_{key}_{hook}'
        t_begin_name = f't_{key}_begin'
        dt_name = f'dt_{key}'

        if hook == 'begin':
            setattr(self, t_begin_name, time.time())

        if hook == 'end':
            t_begin = getattr(self, t_begin_name)
            elapsed_time = time.time() - t_begin
            setattr(self, dt_name, elapsed_time)
            if logs is not None:
                logs.update({dt_name: elapsed_time})

        for callback in self.callbacks:
            step_hook = getattr(callback, hook_name)
            if logs is not None:
                if value is None:
                    step_hook(logs)
                else:
                    step_hook(value, logs)

    def on_step_begin(self, step, logs=None):
        self._call_key_hook('step', 'begin', step , logs)

    def on_step_end(self, step, logs=None):
        self._call_key_hook('step', 'end', step , logs)

    def on_steps_cycle_begin(self, step, logs=None):
        self._call_key_hook('steps_cycle', 'begin', step , logs)

    def on_steps_cycle_end(self, step, logs=None):
        self._call_key_hook('steps_cycle', 'end', step , logs)

    def on_episode_begin(self, episode, logs=None):
        self._call_key_hook('episode', 'begin', episode , logs)

    def on_episode_end(self, episode, logs=None):
        self._call_key_hook('episode', 'end', episode , logs)

    def on_episodes_cycle_begin(self, episode, logs=None):
        self._call_key_hook('episodes_cycle', 'begin', episode , logs)

    def on_episodes_cycle_end(self, episode, logs=None):
        self._call_key_hook('episodes_cycle', 'end', episode , logs)

    def on_run_begin(self, logs=None):
        self._call_key_hook('run', 'begin', logs=logs)

    def on_run_end(self, logs=None):
        self._call_key_hook('run', 'end', logs=logs)
