"""Playground to manage interactions between environment and agent(s)"""

import warnings

from typing import List, Union, Callable, Any
from abc import abstractmethod
from numbers import Number

import numpy as np
from gym import Env

from benchmarks import Agent, TurnEnv
from benchmarks.callbacks import Callback, CallbackList, Logger


class DoneHandler:

    """Helper to modify the done given by the environment.

    You need to specify the method:
     - `done(self, observation, action, reward, done, info, next_observation) -> bool`

    You can also define __init__ and reset() if you want to store anything.

    """

    @abstractmethod
    def done(
        self, observation, action, reward, done, info, next_observation, logs
    ) -> bool:
        """Replace the environment done.

        Often used to make episodes shorter when the agent is stuck for example.

        Args:
            observation: Current observation.
            action: Current action.
            reward: Current reward.
            done: done given by the environment.
            info: Addition informations given by the environment.
            next_observation: Next observation.

        """

    def reset(self):
        """Reset the DoneHandler

        Called automaticaly in :meth:`Playground.run`.
        Used only if variables are stored by the DoneHandler.

        """

    def _done(self, **kwargs) -> bool:
        done = self.done(**kwargs)
        if not isinstance(done, bool) and not (
            isinstance(done, np.ndarray) and done.dtype == bool
        ):
            raise ValueError(
                f"Done should be bool, got {done} of type {type(done)} instead"
            )
        return done

    def __call__(self, **kwargs) -> bool:
        return self._done(**kwargs)


class RewardHandler:

    """Helper to modify the rewards given by the environment.

    You need to specify the method:
     - `reward(self, observation, action, reward, done, info, next_observation) -> float`

    You can also define __init__ and reset() if you want to store anything.

    """

    @abstractmethod
    def reward(
        self, observation, action, reward, done, info, next_observation, logs
    ) -> float:
        """Replace the environment reward.

        Often used to scale rewards or to do reward shaping.

        Args:
            observation: Current observation.
            action: Current action.
            reward: Current reward.
            done: done given by the environment.
            info: Addition informations given by the environment.
            next_observation: Next observation.

        """

    def reset(self):
        """Reset the RewardHandler

        Called automaticaly in :meth:`Playground.run`.
        Useful only if variables are stored by the RewardHandler.

        """

    def _reward(self, **kwargs) -> float:
        reward = self.reward(**kwargs)
        if not isinstance(reward, Number) and not (
            isinstance(reward, np.ndarray) and np.issubdtype(reward.dtype, np.floating)
        ):
            raise ValueError(
                f"Rewards should be a float, got {reward} of type {type(reward)} instead"
            )
        return float(reward)

    def __call__(self, **kwargs) -> float:
        return self._reward(**kwargs)


class Playground:

    """A playground is used to run interactions between an environement and agent(s)

    Attributes:
        env (gym.Env):  Environement in which the agent(s) will play.
        agents (list of learnrl.Agent): List of agents to play.

    """

    def __init__(
        self, environement: Env, agents: Union[Agent, List[Agent]], agents_order=None
    ):
        """A playground is used to run agent(s) on an environement

        Args:
            env: Environement in which the agent(s) will play.
            agents: List of agents to play (can be only one agent).

        """
        if not isinstance(environement, Env):
            raise TypeError("environement should be a subclass of gym.Env")
        if isinstance(agents, Agent):
            agents = [agents]
        for agent in agents:
            if not isinstance(agent, Agent):
                raise TypeError("All agents should be a subclass of learnrl.Agent")

        self.env = environement
        self.agents = agents
        self.agents_order = None
        self.set_agents_order(agents_order)

    @staticmethod
    def _get_episodes_cycle_len(episodes_cycle_len, episodes):
        if 0 < episodes_cycle_len < 1:
            episodes_cycle_len = max(1, int(episodes_cycle_len * episodes))

        episodes_cycle_len = int(episodes_cycle_len)
        if episodes_cycle_len <= 0:
            raise ValueError("episodes_cycle_len must be > 0")

        return episodes_cycle_len

    def _build_callbacks(self, callbacks, logger, params):
        callbacks = callbacks if callbacks is not None else []
        if logger is not None:
            callbacks += [logger]

        callbacks = CallbackList(callbacks)
        callbacks.set_params(params)
        callbacks.set_playground(self)
        return callbacks

    def _reset(self, reward_handler, done_handler):
        previous = [
            {
                "observation": None,
                "action": None,
                "reward": None,
                "done": None,
                "info": None,
            }
            for _ in range(len(self.agents))
        ]
        if isinstance(reward_handler, RewardHandler):
            reward_handler.reset()
        if isinstance(done_handler, DoneHandler):
            done_handler.reset()
        return self.env.reset(), 0, False, previous

    def _get_next_agent(self, observation):
        turn_id = self.env.turn(observation) if isinstance(self.env, TurnEnv) else 0
        try:
            agent_id = self.agents_order[turn_id]
            agent = self.agents[agent_id]
        except IndexError as index_error:
            error_msg = f"Not enough agents to play environement {self.env}"
            raise ValueError(error_msg) from index_error
        return agent, agent_id

    @staticmethod
    def _call_handlers(
        reward, done, experience, reward_handler=None, done_handler=None, logs=None
    ):
        logs = {} if logs is None else logs

        logs.update({"reward": reward})
        if reward_handler is not None:
            handled_reward = reward_handler(**experience, logs=logs)
            experience["reward"] = handled_reward
            logs.update({"handled_reward": handled_reward})

        logs.update({"done": done})
        if done_handler is not None:
            handled_done = done_handler(**experience, logs=logs)
            experience["done"] = handled_done
            logs.update({"handled_done": handled_done})

    def _run_step(
        self,
        observation: Any,
        previous: list,
        logs: dict,
        learn=False,
        render=False,
        render_mode="human",
        reward_handler=None,
        done_handler=None,
    ):
        """Run a single step"""
        # Render the environment
        if render:
            self.env.render(render_mode)

        agent, agent_id = self._get_next_agent(observation)

        # If the agent has played before, perform a learning step
        prev = previous[agent_id]
        if learn and prev["observation"] is not None:
            agent.remember(
                prev["observation"],
                prev["action"],
                prev["reward"],
                prev["done"],
                observation,
                prev["info"],
            )
            agent_logs = agent.learn()
            logs.update({f"agent_{agent_id}": agent_logs})

        # Ask action to agent
        action = agent.act(observation, greedy=not learn)

        # Perform environment step
        next_observation, reward, done, info = self.env.step(action)

        # Adds step informations to logs
        experience = {
            "observation": observation,
            "action": action,
            "reward": reward,
            "done": done,
            "next_observation": next_observation,
            "info": info,
        }
        logs.update({"agent_id": agent_id})
        logs.update(experience)

        # Use Handlers
        self._call_handlers(
            reward, done, experience, reward_handler, done_handler, logs
        )

        # Store experience in prev
        if learn:
            for key, value in zip(prev, [observation, action, reward, done, info]):
                prev[key] = value

            # Perform a last learning step if done
            if done:
                agent.remember(**experience)
                agent_logs = agent.learn()
                logs.update({f"agent_{agent_id}": agent_logs})

        # Do a last rendering if done
        if done and render:
            self.env.render(render_mode)

        return next_observation, done

    def run(
        self,
        episodes: int,
        render: bool = True,
        render_mode: str = "human",
        learn: bool = True,
        steps_cycle_len: int = 10,
        episodes_cycle_len: Union[int, float] = 0.05,
        verbose: int = 0,
        callbacks: List[Callback] = None,
        logger: Callback = None,
        reward_handler: Union[Callable, RewardHandler] = None,
        done_handler: Union[Callable, DoneHandler] = None,
        **kwargs,
    ):
        """Let the agent(s) play on the environement for a number of episodes.

        Additional arguments will be passed to the default logger.

        Args:
            episodes: Number of episodes to run.
            render: If True, call |gym.render| every step.
            render_mode: Rendering mode.
                One of {'human', 'rgb_array', 'ansi'} (see |gym.render|).
            learn: If True, call :meth:`Agent.learn` every step.
            steps_cycle_len: Number of steps that compose a cycle.
            episode_cycle_len: Number of episodes that compose a cycle.
                If between 0 and 1, this in understood as a proportion.
            verbose: The verbosity level: 0 (silent), 1 (cycle), 2 (episode),
                3 (step_cycle), 4 (step), 5 (detailed step).
            callbacks: List of :class:`~learnrl.callbacks.Callback` to use in runs.
            reward_handler: A callable to redifine rewards of the environement.
            done_handler: A callable to redifine the environement end.
            logger: Logging callback to use.
                If None use the default :class:`~learnrl.callbacks.logger.Logger`.

        """
        episodes_cycle_len = self._get_episodes_cycle_len(episodes_cycle_len, episodes)

        params = {
            "episodes": episodes,
            "episodes_cycle_len": episodes_cycle_len,
            "steps_cycle_len": steps_cycle_len,
            "verbose": verbose,
            "render": render,
            "learn": learn,
        }

        logger = logger if logger else Logger(**kwargs)
        callbacks = self._build_callbacks(callbacks, logger, params)

        # Start the run
        logs = {}
        logs.update(params)

        callbacks.on_run_begin(logs)

        for episode in range(episodes):
            if episode % episodes_cycle_len == 0:
                callbacks.on_episodes_cycle_begin(episode, logs)

            observation, step, done, previous = self._reset(
                reward_handler, done_handler
            )

            logs.update({"episode": episode})
            callbacks.on_episode_begin(episode, logs)

            while not done:
                if step % steps_cycle_len == 0:
                    callbacks.on_steps_cycle_begin(step, logs)

                logs.update({"step": step})
                callbacks.on_step_begin(step, logs)

                observation, done = self._run_step(
                    observation=observation,
                    previous=previous,
                    logs=logs,
                    learn=learn,
                    render=render,
                    render_mode=render_mode,
                    reward_handler=reward_handler,
                    done_handler=done_handler,
                )
                callbacks.on_step_end(step, logs)

                if (step + 1) % steps_cycle_len == 0 or done:
                    callbacks.on_steps_cycle_end(step, logs)

                step += 1

            callbacks.on_episode_end(episode, logs)

            if (episode + 1) % episodes_cycle_len == 0 or episode + 1 == episodes:
                callbacks.on_episodes_cycle_end(episode, logs)

        callbacks.on_run_end(logs)

    def fit(self, episodes, **kwargs):
        """Train the agent(s) on the environement for a number of episodes."""
        learn = kwargs.pop("learn", True)
        render = kwargs.pop("render", False)
        if not learn:
            warnings.warn(
                "learn should be True in Playground.fit(), otherwise the agents will not improve",
                UserWarning,
            )
        if render:
            warnings.warn(
                "rendering degrades heavily computation speed", RuntimeWarning
            )

        self.run(episodes, render=render, learn=learn, **kwargs)

    def test(self, episodes, **kwargs):
        """Test the agent(s) on the environement for a number of episodes."""
        learn = kwargs.pop("learn", False)
        render = kwargs.pop("render", True)
        verbose = kwargs.pop("verbose", 0)
        if learn:
            warnings.warn(
                "learn should be False in Playground.test(),"
                "otherwise the agents will not act greedy and can have random behavior",
                UserWarning,
            )
        if not render and verbose == 0:
            warnings.warn(
                "you should set verbose > 0 or render=True to have any feedback ...",
                UserWarning,
            )
        self.run(episodes, render=render, learn=learn, verbose=verbose, **kwargs)

    def set_agents_order(self, agents_order: list) -> list:
        """Change the agents_order.

        This will update the agents order.

        Args:
            agents_order: New agents indices order.
                Default is range(n_agents).

        Returns:
            The updated agents ordered indices list.

        """
        if agents_order is None:
            self.agents_order = list(range(len(self.agents)))
        else:
            if len(agents_order) != len(self.agents):
                raise ValueError(
                    f"Not every agents have an order number.\n"
                    f"Custom order: {agents_order} for {len(self.agents)} agents\n"
                )

            valid_order = True
            for place in range(len(self.agents)):
                if not place in agents_order:
                    valid_order = False

            if not valid_order:
                raise ValueError(
                    f"Custom order is not taking every index in [0, n_agents-1].\n"
                    f"Custom order: {agents_order}"
                )
            self.agents_order = agents_order
        return self.agents_order
