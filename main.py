from learnrl.core import Playground
from learnrl.environments import RdNimEnv, CrossesAndNoughtsEnv
from learnrl.agents import TableAgent
from learnrl.agents.table.control import Greedy
from learnrl.agents.table.evaluation import TemporalDifference
import numpy as np

env = CrossesAndNoughtsEnv()
agent1 = TableAgent(state_space=env.observation_space, action_space=env.action_space)
agent2 = TableAgent(state_space=env.observation_space, action_space=env.action_space)

agents = [agent1, agent2]
pg = Playground(env, agents)
pg.fit(50000, verbose=1)
