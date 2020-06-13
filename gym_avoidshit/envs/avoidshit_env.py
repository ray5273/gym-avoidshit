import gym
import numpy as np
import sys
import copy
import random
import math
import pylab
from time import sleep
from collections import deque
from gym import spaces,error

class AvoidShitEnv(gym.Env):
    metadata = {'render.modes':['human']
            ,'videos.frames_per_second':30}
    
   
    #30 fps
    def __init__(self):
        #Game Variables
        self.PAD_WIDTH = 480
        self.PAD_HEIGHT = 640
        self.RANDOM=False
        #Color Variables
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
 
        #Shit size
        self.ddong_width = 26
        self.ddong_height = 26
        self.total_ddong = 10
        self.ddong_speed = 32
        #Player size
        self.man_width = 36
        self.man_height = 38
        
        #Gym Variables
        self.observation_size = self.total_ddong*2+1
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Discrete(self.observation_size)

        self.reset()


    def step(self, action):
        dx = 0
        if action == -1:
            dx -= 48
        elif action == 1:
            dx += 48
        elif action == 0:
            dx = 0

        self.man_x += dx
        if self.man_x <0:
            self.man_x = 0
        elif self.man_x > self.PAD_WIDTH - self.man_width:
            self.man_x = self.PAD_WIDTH - self.man_width
        
        #move shit
        for index,value in enumerate(self.last_ddong_y):
            self.last_ddong_y[index] += self.ddong_speed
            #if shit touch ground, make new one
            if value > self.PAD_HEIGHT:
                if self.RANDOM:
                    self.last_ddong_x[index] = int(random.randrange(0,self.PAD_WIDTH - self.man_width)/48)*48
                self.last_ddong_y[index] = -self.ddong_height
                self.reward = 1
                self.score +=1
        for index,value in enumerate(self.last_ddong_y):
            if abs(self.last_ddong_x[index] - self.man_x) < self.ddong_width and self.man_y - self.last_ddong_y[index] < self.ddong_height:
                self.done = True
                self.reward = -100
        
        if self.score >= 500:
            self.done = True
        state = self._get_game_state()
        reward = self.reward
        done = self.done
        return state, reward, done, {}

    def reset(self):
        self.board_x,self.board_y = self._fill_board()
        self.man_x = 0
        self.man_y = self.PAD_HEIGHT * 0.9
        self.last_ddong_x = copy.deepcopy(self.board_x)
        self.last_ddong_y = copy.deepcopy(self.board_y)
        self.screen = None
        self.score = 0
        self.reward = 0
        self.done = False
        return self._get_game_state()
         
    def render(self,mode='human',close=False):
        try:
            import pygame
        except ImportError as e :
            raise error.DependencyNotInstalled(
                    "{}. (HINT: install pygame using `pip install pygame`".format(e))
        if close:
            pygame.quit()
        else:
            #update render
            man = pygame.image.load('man.png')
            ddong = pygame.image.load('ddong.png') 
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.PAD_WIDTH,self.PAD_HEIGHT))
            clock = pygame.time.Clock()
            self.screen.fill(self.BLACK)
            self.screen.blit(man,(self.man_x,self.man_y))
            for index,value in enumerate(self.last_ddong_x):
                self.screen.blit(ddong,(self.last_ddong_x[index],self.last_ddong_y[index]))
            font = pygame.font.SysFont(None, 30)
            text = font.render('Score: {}'.format(self.score), True, (255, 255, 255))
            self.screen.blit(text, (380, 30))  
            pygame.display.update() 
            clock.tick(30)

    def close(self):
        pass


    def _fill_board(self):
        ddong_x,ddong_y = [],[]
        fixed_ddong_x = [8, 7, 4, 2, 5, 9, 0, 1, 3, 6, 3, 7, 3, 3, 4, 9, 0, 1, 5, 6, 8, 8, 9, 5, 6, 1, 2, 2, 4, 5]
        fixed_ddong_y = [8, 10, 8, 1, 0, 0, 17, 12, 1, 5, 7, 13, 9, 19, 0, 1, 3, 12, 13, 15, 8, 13, 15, 8, 10, 11, 13, 16, 6, 5]
        for i in range(self.total_ddong):
            ddong_x.append(fixed_ddong_x[i]*48)
            ddong_y.append(fixed_ddong_y[i]*-32)
        
        return ddong_x,ddong_y

    def _get_game_state(self):
        state = self.last_ddong_x + self.last_ddong_y;
        state.append(self.man_x)

        return np.reshape(state,[1,self.observation_size])

