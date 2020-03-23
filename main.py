# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import learnrl as rl
from learnrl.environments import CrossesAndNoughtsEnv
from learnrl.agents import TableAgent

env = CrossesAndNoughtsEnv()
agent1 = TableAgent(state_space=env.observation_space, action_space=env.action_space)
agent2 = TableAgent(state_space=env.observation_space, action_space=env.action_space)

agents = [agent1, agent2]
pg = rl.Playground(env, agents)
pg.fit(50000, verbose=1)
