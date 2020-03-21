from core import Playground
from envs import NimEnv
from agents import BasicAgent
from agents.basic.control import Greedy
from agents.basic.evaluation import TemporalDifference

env = NimEnv(is_optimal=True)
agent = BasicAgent(state_space=env.observation_space, action_space=env.action_space,
                   control=Greedy(env.action_space.n, initial_exploration=0.1, exploration_decay=1-5e-5),
                   evaluation=TemporalDifference(initial_learning_rate=0.2, online=True))

pg = Playground(env, agent)
pg.fit(1000, verbose=2)
pg.run(10)
