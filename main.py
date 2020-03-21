import learnrl as rl
from learnrl.environments import RdNimEnv, CrossesAndNoughtsEnv
from learnrl.agents import TableAgent
from learnrl.agents.table import Greedy, TemporalDifference
import numpy as np

env = CrossesAndNoughtsEnv()
agent1 = TableAgent(state_space=env.observation_space, action_space=env.action_space)
agent2 = TableAgent(state_space=env.observation_space, action_space=env.action_space)

agents = [agent1, agent2]
pg = rl.Playground(env, agents)
pg.fit(50000, verbose=1)
