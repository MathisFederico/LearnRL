import numpy as np

from envs import FrozenLakeEnv
from agents import BasicAgent

if __name__ == "__main__":
    env = FrozenLakeEnv()
    agent = BasicAgent()

    n_games = 1000

    for game in range(n_games):
        done = False
        state = env.reset()
        G = 0
        while not done:
            env.render()
            legal_actions = np.array(range(4))
            action = agent.act(state, legal_actions)
            next_state, reward, done , info = env.step(action)
            G += reward

            agent.remember(state, action, reward, done, next_state, info)
            agent.learn()

            state = next_state
        if G != 0: 
            print('Game {}/{}, Return:{}'.format(game,n_games,G))
            print(agent.action_values)