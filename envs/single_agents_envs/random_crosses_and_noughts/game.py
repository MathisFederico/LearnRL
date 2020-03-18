from gym import Env, spaces
import numpy as np
from copy import deepcopy
import pygame
from pygame.transform import scale
from itertools import product
import os

class CrossesAndNoughtsGame():

    def __init__(self):
        self.grid = np.zeros((3,3), dtype=np.uint8)
        self.crosses = 1
        self.nought = 2
        self.isPygameInit = False

    def is_valid(self, position):
        position = np.array(position)
        if position.ndim == 1:
            position = np.array([position])

        values = []
        for coord in position:
            if np.all(np.array(coord) >= 0) and np.all(np.array(coord) < 3):
                values.append(self.grid[tuple(coord)])
            else: values.append(0)
        
        values = np.array(values, dtype=np.uint8)
        pos = np.array(position)
        return np.logical_and(np.logical_and(np.all(pos >= 0, axis=1), np.all(pos < 3, axis=1)), values == 0)
    
    def getLegalActions(self):
        actions = np.array([(0, 0), (0, 1), (0, 2),
                            (1, 0), (1, 1), (1, 2),
                            (2, 0), (2, 1), (2, 2)])
        return actions[self.is_valid(actions)]
        
    def play(self, player, position):
        move = tuple(position)
        if self.is_valid(move):
            self.grid[move] = player

    def wincheck(self):
        x = np.array([1, 1, 1], dtype=np.uint8)
        o = np.array([2, 2, 2], dtype=np.uint8)
        
        # Check all rows
        for i in range(3):
            if np.all(self.grid[i] == x): return 1
            if np.all(self.grid[i] == o): return 2

        # Check all columns
        for i in range(3):
            t = [self.grid[j, i] for j in range(3)]
            if np.all(t == x): return 1
            if np.all(t == o): return 2

        # Check diagonal
        t = [self.grid[i, i] for i in range(3)]
        if np.all(t == x): return 1
        if np.all(t == o): return 2

        # Check antidiagonal
        t = [self.grid[2 - i, i] for i in range(3)]
        if np.all(t == x): return 1
        if np.all(t == o): return 2
        
        # If no one can play
        if not np.any(self.grid == 0): return 3
        return 0

    def initPygame(self, scale_factor=5):
        # pygame.quit()
        pygame.init() # pylint: disable=E1101 
                
        self.scale_factor = scale_factor
        # Open Pygame window
        self.window = pygame.display.set_mode((200*scale_factor, 100*scale_factor))
        # Load images
        path = os.path.join("envs", "single_agents_envs", "random_crosses_and_noughts")
        self.empty_grid_img = pygame.image.load(os.path.join(path, "images", "empty_grid.png")).convert()
        self.empty_grid_img = scale(self.empty_grid_img, (scale_factor*self.empty_grid_img.get_width(), scale_factor*self.empty_grid_img.get_height()))

        self.black_background = pygame.image.load(os.path.join(path, "images", "black_background.png")).convert()
        self.black_background = scale(self.black_background, (scale_factor*self.black_background.get_width(), scale_factor*self.black_background.get_height()))
        
        cross_img = pygame.image.load(os.path.join(path, "images", "cross.png")).convert()
        cross_img = scale(cross_img, (scale_factor*cross_img.get_width(), scale_factor*cross_img.get_height()))

        nought_img = pygame.image.load(os.path.join(path, "images","nought.png")).convert()
        nought_img = scale(nought_img, (scale_factor*nought_img.get_width(), scale_factor*nought_img.get_height()))

        self.rect_size = scale_factor*cross_img.get_width()

        self.images = {1:cross_img, 2:nought_img}

        self.states_images = {}
        # files = os.listdir(os.path.join(path, 'states'))
        # for filename in files:
        #     state = int(filename.split('.')[0])
        #     img = pygame.image.load(os.path.join(path, "states/{}.png".format(state))).convert()
        #     img = scale(img, (scale_factor*img.get_width(), scale_factor*img.get_height()))
        #     self.states_images[state] = img

        # Refresh display
        self.window.blit(self.empty_grid_img, (0,0))
        pygame.display.flip()
    
    def render(self, frame_limit):

        def getPosition(i, j):
            pos = np.array((34*self.scale_factor*i, 34*self.scale_factor*j), dtype=int)
            return tuple(pos)

        def getNextState(action, nextActionsState): 
            for next_state, actions in nextActionsState.items(): 
                if action in actions: 
                    return next_state 
            return "key doesn't exist"

        if not self.isPygameInit:
            self.initPygame()
            self.isPygameInit = True

        self.window.fill((0, 0, 0))
        nextActionsState = self.getNextActionsState(self.grid)
        states_micro_ids = {next_state:i for (i, next_state) in enumerate(self.getNextActionsState(self.grid))}
        colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255], [64, 64, 64], [192, 192, 192]]

        for i in range(3):
            for j in range(3):
                content = self.grid[(i, j)]
                if content in [1, 2]:
                    image = self.images[content]
                    self.window.blit(image, getPosition(i, j))
                if content == 0:
                    action = (i, j)
                    next_state = getNextState(action, nextActionsState)
                    color = colors[states_micro_ids[next_state]]
                    position = np.array(getPosition(i, j))
                    side_size = 32*self.scale_factor
                    position = tuple(position) + (side_size, side_size)
                    pygame.draw.rect(self.window, color, position)
        
        state = self.getStateHash(self.grid)

        font = pygame.font.Font(None, 12*self.scale_factor)
        text = font.render(str(state), True, (255, 255, 255))
        text_pos = (300*self.scale_factor // 2 - text.get_width() // 2, 100*self.scale_factor // 6 - text.get_height() // 2)
        self.window.blit(self.black_background, (self.scale_factor*100, 0))
        self.window.blit(text, text_pos)

        pygame.display.flip()
        if frame_limit != 0:
            pygame.time.wait(1000//frame_limit)
        
    def getStateHash(self, grid):

        def to_state(grid):
            u = 0
            for i in range(3):
                for j in range(3):
                    u += grid[i][j] * 3**(3*i + j)
            return u

        def flip(grid):
            return grid[::-1]

        def rot(grid):
            flatten = grid[0] + grid[1] + grid[2]
            newflatten = [flatten[i] for i in [2, 5, 8, 1, 4, 7, 0, 3, 6]]
            return [newflatten[3*i:3*i+3] for i in range(3)]

        all_states = []
        tmp = deepcopy(grid.tolist())
        for _ in range(4):
            all_states.append(to_state(tmp))
            tmp = rot(tmp)
        tmp = flip(tmp)
        for _ in range(4):
            all_states.append(to_state(tmp))
            tmp = rot(tmp)
        return min(all_states)
    
    def getNextActionsState(self, grid):

        next_states = {}
        
        nb_X = np.sum(grid==1)
        nb_O = np.sum(grid==2)
        if nb_X == nb_O: player = 1
        else: player = 2
        
        legal_actions = [(i, j) for i in range(3) for j in range(3) if grid[i, j] == 0]

        for action in legal_actions:
            grid_copy = deepcopy(grid)
            grid_copy[action] = player
            next_state_id = self.getStateHash(grid_copy)
            if next_state_id not in next_states:
                next_states[next_state_id] = [action]
            else:
                next_states[next_state_id].append(action)
        return next_states

    def getObservation(self):
        return np.array(self.grid)

    def reset(self):
        self.__init__()


class CrossesAndNoughtsEnv(Env):
    
    def __init__(self, vs_random=False):
        self.game = CrossesAndNoughtsGame()
        self.action_space = spaces.MultiDiscrete((3, 3))
        self.observation_space = spaces.MultiDiscrete(3 * np.ones((3, 3), dtype=np.int64))
        self.vs_random = vs_random

    def step(self, action):

        def checkStep(player, other_player):
            if not pass_turn:
                reward, done = 0, False
                winner = self.game.wincheck()
                if winner == player:
                    reward = 1
                    done = True
                elif winner == other_player:
                    reward = -1
                    done = True
                elif winner == 3:
                    done = True
            else:
                reward, done = -1, False
            return reward, done

        pass_turn = False
        observation = self.game.getObservation()
        nb_X = np.sum(observation==1)
        nb_O = np.sum(observation==2)
        if nb_X == nb_O: player, other_player = 1, 2
        else: player, other_player = 2, 1

        winner = self.game.wincheck()
        if winner == 0:
            if self.game.is_valid(action):
                self.game.play(player, action)
            else:
                # Need play random instead
                legal_actions = self.game.getLegalActions()
                rd_action = np.random.choice(legal_actions)
                self.game.play(player, rd_action)
                print("Invalid action, played at random, reward is -1")
                pass_turn = True
        
        reward, done = checkStep(player, other_player)
        
        if self.vs_random:
            if not done:
                legal_actions = self.game.getLegalActions()
                if len(legal_actions) > 0:
                    rd_action = legal_actions[np.random.choice(range(len(legal_actions)))]
                    self.game.play(other_player, rd_action)
                reward, done = checkStep(other_player, player)

        observation = self.game.getObservation()

        return observation, reward, done, {'pass_turn':pass_turn}
    
    def render(self, frame_limit=0):
        self.game.render(frame_limit=frame_limit)

    def reset(self):
        self.game = CrossesAndNoughtsGame()
        return self.game.getObservation()
    
