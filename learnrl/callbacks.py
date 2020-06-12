import time
import sys

class Callback():

    """ An object to log informations while the :class:`Playground` is running.
    """

    def set_params(self, params):
        self.params = params
    
    def set_playground(self, playground):
        self.playground = playground

    def on_step_begin(self, step, logs=None):
        pass

    def on_step_end(self, step, logs=None):
        pass
    
    def on_episode_begin(self, episode, logs=None):
        pass

    def on_episode_end(self, episode, logs=None):
        pass

    def on_cycle_begin(self, cycle, logs=None):
        pass

    def on_cycle_end(self, cycle, logs=None):
        pass

    def on_run_begin(self, logs=None):
        pass

    def on_run_end(self, logs=None):
        pass

class CallbackList():
    """ An wrapper to use a list of :class:`Callback` while the :class:`~learnrl.core.Playground` is running.
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
            t_begin = getattr(self, t_begin_name, time.time())
            setattr(self, dt_name, time.time() - t_begin)
        
        for callback in self.callbacks:
            step_hook = getattr(callback, hook_name)
            step_hook(logs) if value is None else step_hook(value, logs)
    
    def on_step_begin(self, step, logs=None):
        self._call_key_hook('step', 'begin', step , logs)

    def on_step_end(self, step, logs=None):
        self._call_key_hook('step', 'end', step , logs)
    
    def on_episode_begin(self, episode, logs=None):
        self._call_key_hook('episode', 'begin', episode , logs)

    def on_episode_end(self, episode, logs=None):
        self._call_key_hook('episode', 'end', episode , logs)

    def on_cycle_begin(self, cycle, logs=None):
        self._call_key_hook('cycle', 'begin', cycle , logs)

    def on_cycle_end(self, cycle, logs=None):
        self._call_key_hook('cycle', 'end', cycle , logs)

    def on_run_begin(self, logs=None):
        self._call_key_hook('run', 'begin', logs=logs)

    def on_run_end(self, logs=None):
        self._call_key_hook('run', 'end', logs=logs)

class Logger(Callback):
    pass
    # def __init__(self, verbose):
    #     self.verbose = verbose

    # def on_step_end(self, step, logs=None):
    #     if self.verbose == 3:
    #         print(f"Step: {step} \t| Player {agent_id} \t| Reward {reward}")
    #     if self.verbose > 3:
    #         print(f"------ Step {step} ------ Player is {agent_id}"
    #                 f"\nobservation:\n{observation}\naction:\n{action}\nreward:{reward}\ndone:{done}"
    #                 f"\nnext_observation:\n{next_observation}\ninfo:{info}")
    
    # def on_episode_end(self, episode, logs=None):
    #     if self.verbose == 1:
    #         pass

    # def on_episode_cycle_end(self, cycle, logs=None):
    #     if self.verbose > 0:
    #         steps += step
    #         avg_gain += gain
    #         if episode%print_cycle==0:
    #             dt = max(1e-6, time()-t0)
    #             print(f"Episode {episode}/{episodes}    \t gain:{avg_gain/print_cycle} \t"
    #                     f"steps/s:{steps/dt:.0f}, episodes/s:{print_cycle/dt:.0f}")
    #             avg_gain = np.zeros_like(self.agents)
    #             steps = 0
    #             t0 = time()
