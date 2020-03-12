import numpy as np
from copy import deepcopy

from envs import NimEnv, CrossesAndNoughtsEnv
from agents import BasicAgent
from agents.basic.evaluation import TemporalDifference, MonteCarlo
from agents.basic.control import Greedy

import matplotlib.pyplot as plt

if __name__ == "__main__":
    np.set_printoptions(precision=3)
    env = CrossesAndNoughtsEnv()
    agent = BasicAgent(state_size=env.observation_space.n, action_size=env.action_space.n,
                        control=Greedy(env.action_space.n, initial_exploration=0),
                        evaluation=TemporalDifference(initial_learning_rate=1, online=False))

    n_games = 100 

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
            agent.learn(online=False)

            state = deepcopy(next_state)
            # print(state, action, reward, done, next_state)

        if game%100==0: print('Game {}/{}, Return:{}'.format(game, n_games, G))
    
    action_size, state_size = env.action_space.n, env.observation_space.n
    action_values = agent.action_values

    print(action_values)
    print(1+np.argmax(action_values, axis=1))
    
    X = range(state_size)
    for action in range(action_size):
        plt.plot(X, action_values[:, action], label=1+action, linestyle='-', marker='+')
    plt.legend()
    plt.show()