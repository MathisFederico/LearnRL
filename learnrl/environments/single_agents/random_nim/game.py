# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from gym import Env, spaces
import numpy.random as rd

class NimGame():
    def __init__(self, players=("Player 1", "Player 2"), actions=(1,2,3), nb_sticks=10):
        self.players = players
        self.losers = {self.players[0]:False, self.players[1]:False}
        self.actions = actions
        self.sticks_left = nb_sticks
    
    def is_legal(self, sticks):
        """ Vérifie que l'action est authorisée """
        return sticks in self.actions

    def remove(self, player, sticks):
        """ Enlève les batons aux batons restants si il ne reste plus de batons, vous avez perdu"""
        if not any(lost for lost in self.losers.values()):
            if self.sticks_left < 1:
                player_id = self.players.index(player)
                opponent_id = 1-player_id
                opponent = self.players[opponent_id]
                self.losers[opponent] = 1
            if not any(lost for lost in self.losers.values()):
                self.sticks_left -= sticks
                if self.sticks_left < 1:
                    self.losers[player] = 1

    def won(self, player):
        """ Si l'autre joueur a perdu, c'est que vous avez gagné"""
        return any(lost for lost in self.losers.values()) and self.losers[player]==0

    def get_observation(self):
        """ Return the game state """
        return int(max(self.sticks_left, 0))
    
    def render(self):
        print(self.sticks_left)


class RdNimEnv(Env):

    def __init__(self, initial_state=20, actions=(1,2,3), is_optimal=False):
        self.initial_state = initial_state
        self.players = ("Agent", "Environement")
        self.is_optimal = is_optimal
        self.actions = actions
        self.game = NimGame(players=self.players, actions=self.actions, nb_sticks=self.initial_state)
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Discrete(initial_state + 1)
    
    def legal_actions(self, observation):
        return self.actions

    def step(self, action):
        reward = 0
        done = False
        pass_turn= False

        # Agent turn
        sticks_to_remove_by_agent = self.actions[action]

        if self.game.is_legal(sticks_to_remove_by_agent):
            if self.game.won(self.players[0]):
                reward = 1
                done = True
            else:
                self.game.remove(self.players[0], sticks_to_remove_by_agent)
        else:
            reward = -1
            pass_turn= True

        # Environement turn
        def env_policy(state, actions, random=True):
            if (state - 1)%4==0 or random:
              choice = rd.choice(actions)
            else:
              choice = (state - 1)%4
            return choice
        
        if not pass_turn:
            sticks_to_remove_by_env = env_policy(self.game.get_observation(), self.actions, not self.is_optimal)
            if self.game.won(self.players[1]) and not done:
                reward = -1
                done = True
            else:
                self.game.remove(self.players[1], sticks_to_remove_by_env)
            if self.game.won(self.players[0]) and not done:
                reward = 1
                done = True
        
        observation = self.game.get_observation()
        return observation, reward, done, {}

    def reset(self):
        self.game = NimGame(players=self.players, actions=self.actions,
                            nb_sticks=self.initial_state)
        return self.game.get_observation()

    def render(self, mode='human'):
        self.game.render()

    def close(self):
        pass
