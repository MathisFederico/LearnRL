import numpy as np
from copy import deepcopy

from envs import NimEnv
from agents import BasicAgent
from agents.basic.evaluation import TemporalDifference

import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    env = NimEnv(is_optimal=True)
    agent = BasicAgent(evaluation=TemporalDifference())

    n_games = 2000

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
    action_values = np.array([[agent.action_values[(state,action)] for (state,action) in agent.action_values if action==i] for i in range(3)])
    X = np.array([[state for (state,action) in agent.action_values if action==i] for i in range(3)])
    print(action_values)
    print(1+np.argmax(action_values, axis=0))
    for action in range(3):
        plt.plot(X[action, :], action_values[action, :], label=1+action, linestyle='', marker='+')
    plt.show()