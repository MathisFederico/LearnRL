import numpy as np
from copy import deepcopy

from envs import NimEnv, CrossesAndNoughtsEnv, FrozenLakeEnv
from agents import BasicAgent
from agents.basic.evaluation import TemporalDifference, MonteCarlo
from agents.basic.control import Greedy, UCB, Puct

import matplotlib.pyplot as plt

if __name__ == "__main__":
    env = CrossesAndNoughtsEnv(vs_random=True, play_first=True)
    state_size = np.prod(env.observation_space.nvec)
    action_size = np.prod(env.action_space.nvec)
    agent = BasicAgent(state_space=env.observation_space, action_space=env.action_space,
                        control=Greedy(action_size, initial_exploration=0.5, exploration_decay=1-5e-5),
                        evaluation=TemporalDifference(initial_learning_rate=0.2, online=True))

    n_games = 10000
    G = 0.0
    steps_per_ep = 0
    for game in range(n_games):
        done = False
        state = env.reset()
        while not done:
            steps_per_ep += 1
            # env.render(frame_limit=0)
            legal_actions = env.game.getLegalActions()
            action = agent.act(state, legal_actions)
            next_state, reward, done, info = env.step(action)
            # if reward != 0 : print(reward)
            # if done: print(reward)
            G += reward

            agent.remember(state, action, reward, done, next_state, info)
            agent.learn()

            state = deepcopy(next_state)

        if game%100==0: 
            print(f'Game {game}/{n_games}, Average Return:{G/100:.3f}, Average Steps per episode :{steps_per_ep/100:.1f}, Exploration:{agent.control.exploration:.2f}')
            G, steps_per_ep = 0, 0
    
    action_values = agent.action_values

    print(action_values)
    print(1+np.argmax(action_values, axis=1))
    
    X = range(state_size)
    for action in range(action_size):
        plt.plot(X, action_values[:, action], label=1+action, linestyle='-', marker='+')
    plt.legend()
    plt.show()