import gym
import numpy as np
import random
import math
from gym import spaces, error

class enemy:
    def __init__(self, side, width, height):
        # 0:up, 1:down, 2:left, 3:right (from u,d,l,r)
        self.width = width
        self.height = height
        self.curve = 0
        if random.random() < 1/10 :
            self.curve = 1
        elif random.random() < 1/9 :
            self.curve = -1
        self.x = random.randrange(0,width)
        self.y = random.randrange(0,height)
        self.rad = math.pi/180
        self.angle = 0.
        if side == 0:
            self.y = 0
            self.angle = random.uniform(0, 180)
        elif side == 1:
            self.y = height
            self.angle = random.uniform(180, 360)
        elif side == 2:
            self.x = 0
            self.angle = random.uniform(-90, 90)
        elif side == 3:
            self.x = width
            self.angle = random.uniform(90, 270)
        self.dir_x = math.cos(self.angle * self.rad)
        self.dir_y = math.cos(self.angle * self.rad)
        self.speed = random.uniform(3,6)

    # move 1 timestep
    def move(self):
        self.angle += self.curve * random.uniform(-self.speed/4, self.speed/2)
        dx = self.speed * math.cos(self.angle * self.rad)
        dy = self.speed * math.sin(self.angle * self.rad)
        self.x = (self.x + dx) % self.width
        self.y = (self.y + dy) % self.height

    def getxy(self):
        return self.x, self.y

class Dodge(gym.Env):
    metadata = {'render.modes': ['human']
        , 'videos.frames_per_second': 60}

    # 60 fps
    def __init__(self):
        # Game Variables
        self.PAD_WIDTH = 200
        self.PAD_HEIGHT = 200
        self.MAN_SIZE = 10
        self.ENEMY_SIZE = 10
        self.ENEMY_NUM = 10
        self.SPEED = 5  # Player Speed

        self.USE_RENDER = False
        self.FIX_ENV = True

        # Gym Variables
        self.observation_size = 2 + self.ENEMY_NUM * 2
        low = np.zeros(self.observation_size)
        high = np.ones(self.observation_size)
        self.action_space = spaces.Box(
            low = 0.,
            high = 1., shape=(1,),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low = low,
            high = high,
            dtype=np.float32
        )
        print(self.observation_space.shape)

        self.reset()

    # action : angle
    def step(self, action):
        self.score += 1
        angle = 2 * action * math.pi
        dx = self.SPEED * math.cos(angle)
        dy = self.SPEED * math.sin(angle)

        self.man_x += dx
        self.man_y += dy

        if self.man_x < 0:
            self.man_x = 0
        elif self.man_x > self.PAD_WIDTH - self.MAN_SIZE:
            self.man_x = self.PAD_WIDTH - self.MAN_SIZE
        if self.man_y < 0:
            self.man_y = 0
        elif self.man_y > self.PAD_HEIGHT - self.MAN_SIZE:
            self.man_y = self.PAD_HEIGHT - self.MAN_SIZE

        for enemy in self.enemies:
            enemy.move()

        if self.score >= 3000:
            self.done = True

        reward = 1
        if self.check_crash():
            self.done = True
            reward = -1

        state = self._get_game_state()
        done = self.done
        score = self.score
        return state, reward, done, score

    def reset(self):
        if self.FIX_ENV :
            seed = random.randrange(10)
            random.seed(seed)
        self.man_x = self.PAD_WIDTH/2
        self.man_y = self.PAD_HEIGHT/2
        self.enemies = []
        for _ in range(self.ENEMY_NUM):
            self.enemies.append(enemy(random.randrange(0, 4), self.PAD_WIDTH, self.PAD_HEIGHT))

        self.screen = None
        self.score = 0
        self.done = False
        return self._get_game_state()


    def render(self, mode='human', close=False):
        try:
            import pygame
        except ImportError as e:
            raise error.DependencyNotInstalled(
                "{}. (HINT: install pygame using `pip install pygame`".format(e))
        def render_init():
            man_img = pygame.image.load('man.png')
            self.man_img = pygame.transform.scale(man_img, (self.MAN_SIZE, self.MAN_SIZE))
            enemy_img = pygame.image.load('enemy.png')
            self.enemy_img = pygame.transform.scale(enemy_img, (self.ENEMY_SIZE, self.ENEMY_SIZE))
        if not self.USE_RENDER :
            render_init()
        if close:
            pygame.quit()
        else:
            # update render
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.PAD_WIDTH, self.PAD_HEIGHT))
            clock = pygame.time.Clock()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.FIX_ENV = not self.FIX_ENV
                        if self.FIX_ENV : random.seed()
                        print("Fixed env : "+str(self.FIX_ENV))

            self.screen.fill((255, 255, 255))
            self.screen.blit(self.man_img, (self.man_x, self.man_y))
            for enemy in self.enemies:
                self.screen.blit(self.enemy_img, enemy.getxy())
            font = pygame.font.SysFont(None, 25)
            text = font.render('Score: {}'.format(self.score), True, (0, 0, 0))
            self.screen.blit(text, (self.PAD_WIDTH/2, 30))
            pygame.display.update()
            clock.tick(60)

    def close(self):
        pass

    # True : Crashed
    def check_crash(self):
        for enemy in self.enemies:
            # crash condition : man_x - ENEMY_SIZE < enemy.x < man_x + MAN_SIZE. (y is same)
            if (enemy.x > self.man_x - self.ENEMY_SIZE and
                    enemy.x < self.man_x + self.MAN_SIZE and
                    enemy.y > self.man_y - self.ENEMY_SIZE and
                    enemy.y < self.man_y + self.MAN_SIZE):
                return True
        return False

    def _get_game_state(self):
        state = []
        for enemy in self.enemies :
            x, y = enemy.getxy()
            state.append(x/self.PAD_WIDTH)
            state.append(y/self.PAD_HEIGHT)
        state.append(self.man_x/self.PAD_WIDTH)
        state.append(self.man_y/self.PAD_HEIGHT)

        return np.reshape(state, [1, self.observation_size])
