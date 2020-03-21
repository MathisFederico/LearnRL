import numpy as np
from copy import deepcopy


from envs import NimEnv, CrossesAndNoughtsEnv, FrozenLakeEnv
from agents import BasicAgent
from agents.basic.evaluation import TemporalDifference, MonteCarlo
from agents.basic.control import Greedy, UCB, Puct

import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = CrossesAndNoughtsEnv(vs_random=True)
    state_size = np.prod(env.observation_space.nvec)
    action_size = np.prod(env.action_space.nvec)
    # agent = BasicAgent(state_space=env.observation_space, action_space=env.action_space,
    #                     control=Greedy(action_size, initial_exploration=0.5, exploration_decay=0.99),
    #                     evaluation=TemporalDifference(initial_learning_rate=0.2, online=False))
    agent = BasicAgent(state_space=env.observation_space, action_space=env.action_space,
                           initial_learning_rate=0.3, initial_exploration=0.1)

    n_games = 10000
    G = 0.0

    for game in range(n_games):
        done = False
        state = env.reset()
        while not done:
            # env.render(frame_limit=1)
            legal_actions = env.game.getLegalActions()
            action = agent.act(state, legal_actions)
            next_state, reward, done, info = env.step(action)
            # print(reward)
            # if done: print('----------\n')

            G += reward

            agent.remember(state, action, reward, done, next_state, info)
            agent.learn()

            state = deepcopy(next_state)

        if game%100==0: 
            print(f'Game {game}/{n_games}, Average Return:{G/100:.3f}')
            G = 0.0
    
    action_values = agent.action_values

    print(action_values)
    print(1+np.argmax(action_values, axis=1))
    
    X = range(state_size)
    for action in range(action_size):
        plt.plot(X, action_values[:, action], label=1+action, linestyle='-', marker='+')
    plt.legend()
    plt.show()