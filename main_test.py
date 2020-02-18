from envs import FrozenLakeEnv
from agents import BasicAgent

if __name__ == "__main__":
    env = FrozenLakeEnv()
    agent = BasicAgent()

    done = False
    state = env.reset()

    while not done:
        action = agent.act()
