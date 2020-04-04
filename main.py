# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import learnrl as rl
from learnrl.environments import RdNimEnv, CrossesAndNoughtsEnv
from learnrl.agents.deepRL.agent import DeepRLAgent
from learnrl.agents import TableAgent

env = CrossesAndNoughtsEnv()
agent = DeepRLAgent(observation_space=env.observation_space, action_space=env.action_space)

agents = [agent, agent]
pg = rl.Playground(env, agents)
pg.run(10000, render=False, verbose=1)
