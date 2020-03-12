import numpy as np
from copy import deepcopy

from envs import NimEnv, CrossesAndNoughtsEnv
from agents import BasicAgent
from agents.basic.evaluation import TemporalDifference

import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    env = CrossesAndNoughtsEnv()
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

        print('Game {}/{}, Return:{}'.format(game, n_games, G))
    
    action_size, state_size = np.prod(env.action_space.nvec), np.prod(env.observation_space.nvec)
    action_values = np.zeros((action_size,state_size))
    for action in range(action_size):
        for state in range(state_size):
            try:
                action_values[action, state] = agent.action_values[(state, action)]
            except KeyError:
                print(f'Unseen couple (state,action) : {(state, action)}')

    print(action_values)
    print(1+np.argmax(action_values, axis=0))
    
    X = range(state_size)
    for action in range(action_size):
        plt.plot(X, action_values[action, :], label=1+action, linestyle='-', marker='+')
    plt.legend()
    plt.show()