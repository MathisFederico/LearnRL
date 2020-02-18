import numpy as np

from envs import FrozenLakeEnv
from agents import BasicAgent

if __name__ == "__main__":
    env = FrozenLakeEnv()
    agent = BasicAgent()

    done = False
    state = env.reset()

    while not done:
        legal_actions = np.array(range(4))
        action = agent.act(state, legal_actions)
        next_state, reward, done , info = env.step(action)
        agent.remember(state, action, reward, done, next_state, info)
    agent.learn()