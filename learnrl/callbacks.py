import time
import sys
import numpy as np

class Callback():

    """ An object to call functions while the :class:`~learnrl.core.Playground` is running.
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
        
    def _call_key_hook(self, key, hook, value=None, logs={}):
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
            logs.update({dt_name: dt})
        
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

    """ Default logger in every :class:`~learnrl.core.Playground` run """

    bar_lenght = 40

    def on_step_begin(self, step, logs=None):
        if self.params['verbose'] == 3:
            print(f'Step {step+1}', end='\t| ')
        elif self.params['verbose'] > 3:
            text = f'Step {step+1}'
            semibar_lenght = (self.bar_lenght - len(text)) // 2 - 1
            odd = (self.bar_lenght - len(text)) % 2 == 1
            print('-' * semibar_lenght + f' {text} ' + '-' * (semibar_lenght + 1*odd))

    def on_step_end(self, step, logs=None):
        self.episode_seen_steps += 1
        self.cycle_seen_steps += 1

        agent_id = logs.get('agent_id')
        reward = logs.get('reward')
        done = logs.get('done')
        self.returns[agent_id] += reward

        step_time = logs.get('dt_step')
        self.episode_step_time += (step_time - self.episode_step_time) / self.episode_seen_steps
        self.cycle_step_time += (step_time - self.cycle_step_time) / self.cycle_seen_steps
        
        if self.params['verbose'] == 3:
            if self.n_agents > 1:
                print(f"Agent :{logs.get('agent_id')}", end='\t| ')
            print(f"Reward {reward}", end='\t| ')
            print(self._get_time_text(step_time, 'step'))
            if done:
                print()
        elif self.params['verbose'] > 3:
            if self.n_agents > 1:
                print(f"Agent {agent_id}")
            print(f"Observation {logs.get('observation')}")
            print(f"Action {logs.get('action')}")
            print(f"Reward {reward}")
            print(f"Done {done}")
            print(f"Next Observation {logs.get('next_observation')}")
            print(self._get_time_text(step_time, 'step'))
            print('-'*self.bar_lenght, end='\n\n')

    def on_episode_begin(self, episode, logs=None):
        self.returns = np.zeros(self.n_agents)
        self.episode_step_time = 0
        self.episode_seen_steps = 0
        if self.params['verbose'] > 2:
            print("="*self.bar_lenght)
        if self.params['verbose'] >= 2:
            print("Episode " + self._get_episode_text(episode), end=' | ')
        if self.params['verbose'] > 2:
            print()

    def on_episode_end(self, episode, logs=None):
        self.cycle_seen_episodes += 1
        self.avg_returns += (self.returns - self.avg_returns) / self.cycle_seen_episodes

        episode_time = logs.get('dt_episode')
        self.cycle_episode_time += (episode_time - self.cycle_episode_time) / self.cycle_seen_episodes

        if self.params['verbose'] >= 2:
            print(f"Returns {self.returns}", end='\t| ')
            print(self._get_time_text(episode_time, 'episode'), end='\t| ')
            print(self._get_time_text(self.episode_step_time, 'step'))
        if self.params['verbose'] > 2:
            print("="*self.bar_lenght, end='\n\n')

    def on_cycle_begin(self, episode, logs=None):
        self.avg_returns = np.zeros(self.n_agents)

        self.cycle_seen_episodes = 0
        self.cycle_episode_time = 0

        self.cycle_seen_steps = 0
        self.cycle_step_time = 0

    def on_cycle_end(self, episode, logs=None):
        if self.params['verbose'] == 1:
            print("Episode " + self._get_episode_text(episode), end=' | ')
            print(f"Returns {self.avg_returns}", end='\t| ')
            print(self._get_time_text(self.cycle_episode_time, 'episode'), end='\t| ')
            print(self._get_time_text(self.cycle_step_time, 'step'))

    def on_run_begin(self, logs=None):
        self.n_agents = len(self.playground.agents)
        self.n_digits_episodes = int(np.log10(self.params['episodes'])) + 1
        if self.params['verbose'] >= 1:
            print('Run started')

    def on_run_end(self, logs=None):
        pass

    def _get_episode_text(self, episode):
        text = f"{episode+1}"
        text = " "*(self.n_digits_episodes - len(text)) + text
        text += f"/{self.params['episodes']}"
        return text
    
    def _get_time_text(self, dt, unit):
        if dt < 1e-9:
            return f'Instant'
        if dt < 1e-6:
            return f'{dt/1e-9:.01f}ns/{unit}'
        if dt < 1e-3:
            return f'{dt/1e-6:.01f}us/{unit}'
        if dt < 1:
            return f'{dt/1e-3:.01f}ms/{unit}'
        return f'{dt:.01f}s/{unit}'

