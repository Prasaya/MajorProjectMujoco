from enum import Enum
import gym
from gym import spaces
from stable_baselines3 import DQN
import numpy as np
from stable_baselines3.common.env_checker import check_env
import pygame
from pygame.locals import *


import numpy as np
import gym
from gym import spaces


class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, maze):
        super(MazeEnv, self).__init__()

        self.rng = np.random.default_rng(0)

        self.maze = np.array(maze)
        self.shape = self.maze.shape
        print('Maze shape', self.shape)

        self.range_x = (0, self.shape[0])
        self.range_y = (0, self.shape[1])

        self.start = (100, 100)
        self.goal = (self.rng.integers(self.range_x[0], self.range_x[1]),
                     self.rng.integers(self.range_y[0], self.range_y[1]))
        self.goal = (395, 175)
        self.current_pos = self.start

        # Four possible actions: up, down, left, right
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
            'position_x': spaces.Discrete(self.shape[0]),
            'position_y': spaces.Discrete(self.shape[1]),
            'goal_x': spaces.Discrete(self.shape[0]),
            'goal_y': spaces.Discrete(self.shape[1]),
            'surroundings': spaces.Box(low=0, high=1, shape=(4,), dtype=np.int16),
        })
        self.reward_range = (-1, 1)

        self.block_size = 5  # Size of each block in the maze
        # Width of the Pygame window
        self.width = self.block_size * self.shape[0]
        # Height of the Pygame window
        self.height = self.block_size * self.shape[1]
        self.wall_color = (0, 0, 0)  # Color of the walls
        self.free_color = (255, 255, 255)  # Color of the free cells
        self.start_color = (0, 255, 0)  # Color of the starting cell
        self.goal_color = (255, 0, 0)  # Color of the goal cell
        self.agent_color = (0, 0, 255)  # Color of the agent

        self.screen = None
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))

    def possible_movements(self):
        x, y = self.current_pos

        movement_possible = {
            Actions.UP: y > 0 and self.maze[x, y-1] != 1,
            Actions.DOWN: y < self.maze.shape[1]-1 and self.maze[x, y+1] != 1,
            Actions.LEFT: x > 0 and self.maze[x-1, y] != 1,
            Actions.RIGHT: x < self.maze.shape[0]-1 and self.maze[x+1, y] != 1,
        }

        return [k for k, v in movement_possible.items() if v]

    def move(self, action: Actions):
        x, y = self.current_pos
        if action == Actions.UP:
            y = max(y-1, 0)
        elif action == Actions.DOWN:
            y = min(y+1, self.maze.shape[1]-1)
        elif action == Actions.LEFT:
            x = max(x-1, 0)
        elif action == Actions.RIGHT:
            x = min(x+1, self.maze.shape[0]-1)
        self.current_pos = (x, y)
        return self.current_pos

    def step(self, action):
        action = Actions(action)
        reward = 0

        allowed_movements = self.possible_movements()
        if action in allowed_movements:
            reward = -0.1
            self.move(action)
        else:
            reward = -1

        if self.current_pos == self.goal:
            reward = 1

        done = self.current_pos == self.goal or len(allowed_movements) == 0

        if done:
            self.reset()

        observation = {
            'position_x': int(self.current_pos[0]),
            'position_y': int(self.current_pos[1]),
            'goal_x': int(self.goal[0]),
            'goal_y': int(self.goal[1]),
            'surroundings': self.get_surroundings(),
        }
        return observation, reward, done, {}

    def get_surroundings(self):
        row, col = self.current_pos
        surroundings = np.zeros(4, dtype=np.int16)
        surroundings[0] = self.maze[row-1, col] if row > 0 else 1
        surroundings[1] = self.maze[row+1,
                                    col] if row < self.maze.shape[0]-1 else 1
        surroundings[2] = self.maze[row, col-1] if col > 0 else 1
        surroundings[3] = self.maze[row, col +
                                    1] if col < self.maze.shape[1]-1 else 1
        return surroundings

    def reset(self):
        x, y = self.rng.integers(self.range_x[0], self.range_x[1]), self.rng.integers(
            self.range_y[0], self.range_y[1])
        self.current_pos = self.start
        x, y = self.rng.integers(self.range_x[0], self.range_x[1]), self.rng.integers(
            self.range_y[0], self.range_y[1]
        )
        self.goal = (x, y)
        observation = {
            'position_x': int(self.current_pos[0]),
            'position_y': int(self.current_pos[1]),
            'goal_x': int(self.goal[0]),
            'goal_y': int(self.goal[1]),
            'surroundings': self.get_surroundings(),
        }
        return observation

    def render(self, mode='human'):
        if mode == 'human':
            # print('Current position:', self.current_pos)

            self.screen.fill(self.free_color)

            for x in range(self.maze.shape[0]):
                for y in range(self.maze.shape[1]):
                    pos_x = x * self.block_size
                    pos_y = y * self.block_size
                    rect = pygame.Rect(
                        pos_x, pos_y, self.block_size, self.block_size)
                    if self.maze[x, y] == 1:
                        pygame.draw.rect(self.screen, self.wall_color, rect)
                    elif (x, y) == self.goal:
                        pygame.draw.rect(self.screen, self.goal_color, rect)
            x = self.current_pos[0] * self.block_size
            y = self.current_pos[1] * self.block_size
            rect = pygame.Rect(x, y, self.block_size, self.block_size)
            pygame.draw.rect(self.screen, self.agent_color, rect)
            pygame.display.flip()
        else:
            super(MazeEnv, self).render(mode=mode)

    def application(self):
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    return

            self.render()
