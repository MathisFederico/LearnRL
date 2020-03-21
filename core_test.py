from core import Playground
from envs import NimEnv, CrossesAndNoughtsEnv
from agents import BasicAgent
from agents.basic.control import Greedy
from agents.basic.evaluation import TemporalDifference
import numpy as np

env = CrossesAndNoughtsEnv()
action_size = np.prod(env.action_space.nvec)
agent = BasicAgent(state_space=env.observation_space, action_space=env.action_space,
                   control=Greedy(action_size, initial_exploration=0.1, exploration_decay=1-5e-5),
                   evaluation=TemporalDifference(initial_learning_rate=0.2, online=True))

pg = Playground(env, agent)
pg.fit(20000, verbose=2)
pg.run(10)
