# LearnRL a python library to learn and use reinforcement learning
# Copyright (C) 2020 Mathïs FEDERICO <https://www.gnu.org/licenses/>

from gym import Env, spaces
import numpy.random as rd
import numpy as np

class CatchEnv(Env):

    def __init__(self, area_shape=(1, 1), n_apples=10, n_steps=None,
                       basket_shape=(0.2, 0.05), apple_shape=(0.05, 0.05),
                       basket_speed=0.05, apple_speed=0.1):
        self.action_space = spaces.Discrete(3)

        self.max_observation = (
            area_shape[0] - apple_shape[0],
            area_shape[1] - apple_shape[1],
            area_shape[0] - basket_shape[0]
        )
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]),
                                            high=np.array(self.max_observation),
                                            dtype=np.float64)

        self.area_shape = area_shape

        self.basket_shape = basket_shape
        self.basket_speed = basket_speed

        self.apple_shape = apple_shape
        self.apple_speed = apple_speed

        self.n_steps = int(n_apples * (area_shape[1]/apple_speed)) if n_steps is None else n_steps
        self.steps = 0
        self.n_apples = n_apples
        self.apples = 0

        self._get_new_apple()
        self.basket_pos = self.max_observation[2] / 2

    def step(self, action):
        self._update_basket_pos(action)
        reward = self._update_apple_pos()
        observation = self._get_observation()
        done = self.steps > self.n_steps or self.apples > self.n_apples
        self.steps += 1
        return observation, reward, done, {}
    
    def reset(self):
        self.apples = 0
        self.steps = 0
        self._get_new_apple()
        self.basket_pos = self.max_observation[2] / 2
        return self._get_observation()
    
    def _update_apple_pos(self):
        self.apple_pos[1] -= self.apple_speed
        if (self.apple_pos[1] < self.basket_shape[1] and
            self.apple_pos[0] < self.basket_pos + self.basket_shape[0] and
            self.apple_pos[0] > self.basket_pos):
            self._get_new_apple()
            return 1 / self.n_apples
        elif self.apple_pos[1] <= 0:
            self._get_new_apple()
            return -1 / self.n_apples
        return 0

    def _update_basket_pos(self, action):
        if action == 1:
            self.basket_pos -= self.basket_speed
        elif action == 2:
            self.basket_pos += self.basket_speed
        self.basket_pos = np.clip(self.basket_pos, 0, self.max_observation[2])
    
    def _get_observation(self):
        return np.array(self.apple_pos + [self.basket_pos])
    
    def _get_new_apple(self):
        self.apple_pos = [rd.rand()*self.max_observation[0], self.max_observation[1]]
        self.apples += 1

    def render(self):
        pass
    
