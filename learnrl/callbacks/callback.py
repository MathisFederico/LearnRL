# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

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

    def set_params(self, params):
        self.params = params
    
    def set_playground(self, playground):
        self.playground = playground

    def on_step_begin(self, step, logs=None):
        pass

    def on_step_end(self, step, logs=None):
        pass
    
    def on_steps_cycle_begin(self, episode, logs=None):
        pass

    def on_steps_cycle_end(self, episode, logs=None):
        pass

    def on_episode_begin(self, episode, logs=None):
        pass

    def on_episode_end(self, episode, logs=None):
        pass

    def on_episodes_cycle_begin(self, episode, logs=None):
        pass

    def on_episodes_cycle_end(self, episode, logs=None):
        pass

    def on_run_begin(self, logs=None):
        pass

    def on_run_end(self, logs=None):
        pass


class CallbackList():
    """ An wrapper to use a list of :class:`Callback` while the :class:`~learnrl.playground.Playground` is running.
    """
    
    def __init__(self, callbacks=[]):
        self.callbacks = callbacks
        self.params = {}
        self.playground = None
    
    def set_params(self, params):
        self.params = params
        for callback in self.callbacks:
            callback.set_params(params)
    
    def set_playground(self, playground):
        self.playground = playground
        for callback in self.callbacks:
            callback.set_playground(playground)
        
    def _call_key_hook(self, key, hook, value=None, logs=None):
        """ Helper func for {step|episode|cycle|run}_{begin|end} methods. """

        if len(self.callbacks) == 0:
            return
        
        hook_name = f'on_{key}_{hook}'
        t_begin_name = f't_{key}_begin'
        dt_name = f'dt_{key}'

        if hook == 'begin':
            setattr(self, t_begin_name, time.time())
        
        if hook == 'end':
            t_begin = getattr(self, t_begin_name)
            dt = time.time() - t_begin
            setattr(self, dt_name, dt)
            if logs is not None:
                logs.update({dt_name: dt})
        
        for callback in self.callbacks:
            step_hook = getattr(callback, hook_name)
            if logs is not None:
                step_hook(logs) if value is None else step_hook(value, logs)
    
    def on_step_begin(self, step, logs=None):
        self._call_key_hook('step', 'begin', step , logs)

    def on_step_end(self, step, logs=None):
        self._call_key_hook('step', 'end', step , logs)

    def on_steps_cycle_begin(self, episode, logs=None):
        self._call_key_hook('steps_cycle', 'begin', episode , logs)

    def on_steps_cycle_end(self, episode, logs=None):
        self._call_key_hook('steps_cycle', 'end', episode , logs)

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
