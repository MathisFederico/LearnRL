from core import Playground
from envs import NimEnv
from agents import BasicAgent

env = NimEnv()
agent = BasicAgent(env.observation_space.n, env.action_space.n)
pg = Playground(env, agent)

pg.fit(1000, verbose=1)
