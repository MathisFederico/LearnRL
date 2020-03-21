from core import Playground
from envs import RdNimEnv, CrossesAndNoughtsEnv
from agents import BasicAgent
from agents.basic.control import Greedy
from agents.basic.evaluation import TemporalDifference
import numpy as np

env = CrossesAndNoughtsEnv()
action_size = np.prod(env.action_space.nvec)

agent1 = BasicAgent(state_space=env.observation_space, action_space=env.action_space,
                   control=Greedy(action_size, initial_exploration=0.3, exploration_decay=1-5e-6),
                   evaluation=TemporalDifference(initial_learning_rate=0.2, online=True))

agent2 = BasicAgent(state_space=env.observation_space, action_space=env.action_space,
                   control=Greedy(action_size, initial_exploration=0.3, exploration_decay=1-5e-6),
                   evaluation=TemporalDifference(initial_learning_rate=0.2, online=True))

agents = [agent1, agent2]
pg = Playground(env, agents)
pg.fit(100000, verbose=1)
