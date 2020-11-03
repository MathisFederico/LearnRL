# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import pytest
import numpy as np
np.random.seed(42)

from copy import deepcopy

from learnrl import Playground, Memory

from learnrl.environments import RdNimEnv
from learnrl.agents import StandardAgent
from learnrl.evaluation import MonteCarlo, TemporalDifference, QLearning
from learnrl.control import Greedy, Ucb, Puct


@pytest.fixture
def n_triplets():
    return 3

@pytest.fixture
def env(n_triplets):
    return RdNimEnv(initial_state=n_triplets*3 + 2, is_optimal=True)

@pytest.fixture
def pertinent_observations(n_triplets):
    return np.concatenate([[2+k+4*i for k in range(3)] for i in range(n_triplets-1)])

@pytest.fixture
def expected_optimal_actions(n_triplets):
    return np.concatenate([(0, 1, 2) for _ in range(n_triplets-1)])

def test_evalutations(env, pertinent_observations, expected_optimal_actions):
    evaluations = [
        (QLearning(learning_rate=0.3), True),
        (QLearning(learning_rate=0.3), False),
        (TemporalDifference(learning_rate=0.3), True),
        (TemporalDifference(learning_rate=0.3), False),
        (MonteCarlo(learning_rate=0.3), False),
    ]
    for evaluation, online in evaluations:
        agent = StandardAgent(observation_space=env.observation_space, action_space=env.action_space,
                        evaluation=evaluation, online=online,
                        control=Greedy(exploration=0.))
        n_games = 50
        pg = Playground(env, agent)
        pg.fit(n_games, verbose=1)

        n_sticks = env.observation_space.n
        greedy_actions = np.zeros(n_sticks)
        for i in range(n_sticks):
            greedy_actions[i] = agent.act(i, greedy=True)
        pertinent_actions = greedy_actions[pertinent_observations]

        if not np.all(pertinent_actions==expected_optimal_actions):
            raise ValueError(f'Got policy {pertinent_actions} instead of {expected_optimal_actions} for evaluation {str(evaluation)}')

def test_controls(env, pertinent_observations, expected_optimal_actions):
    controls = [
        Greedy(exploration=0.1),
        Ucb(exploration=0.1),
        Puct(exploration=0.05),
    ]
    for control in controls:
        agent = StandardAgent(observation_space=env.observation_space, action_space=env.action_space,
                        evaluation=QLearning(learning_rate=0.3), online=True, control=control)
        n_games = 50
        pg = Playground(env, agent)
        pg.fit(n_games, verbose=1)

        n_sticks = env.observation_space.n
        greedy_actions = np.zeros(n_sticks)
        for i in range(n_sticks):
            greedy_actions[i] = agent.act(i, greedy=True)
        pertinent_actions = greedy_actions[pertinent_observations]

        if not np.all(pertinent_actions==expected_optimal_actions):
            raise ValueError(f'Got policy {pertinent_actions} instead of {expected_optimal_actions} for control {str(control)}')
