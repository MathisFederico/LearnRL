from core import Playground
from envs import RdNimEnv, CrossesAndNoughtsEnv
from agents import BasicAgent
from agents.basic.control import Greedy
from agents.basic.evaluation import TemporalDifference
import numpy as np

env = CrossesAndNoughtsEnv()
agent1 = BasicAgent(state_space=env.observation_space, action_space=env.action_space)
agent2 = BasicAgent(state_space=env.observation_space, action_space=env.action_space)

agents = [agent1, agent2]
pg = Playground(env, agents)
pg.fit(50000, verbose=1)
