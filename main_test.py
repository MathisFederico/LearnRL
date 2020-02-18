import numpy as np
from copy import deepcopy

from envs import NimEnv
from agents import BasicAgent

if __name__ == "__main__":
    np.set_printoptions(precision=2, suppress=True)
    env = NimEnv(is_optimal=True)
    agent = BasicAgent()

    n_games = 1000

    for game in range(n_games):
        done = False
        state = env.reset()
        G = 0
        while not done:
            # env.render()
            legal_actions = np.array(range(3))
            action = agent.act(state, legal_actions)
            next_state, reward, done , info = env.step(action)
            G += reward

            agent.remember(state, action, reward, done, next_state, info)
            # print(len(agent.memory.datas['state']), done)
            agent.learn()

            state = deepcopy(next_state)
            # print(state, action, reward, done, next_state)

        print('Game {}/{}, Return:{}'.format(game,n_games,G))
    print(agent.action_values)