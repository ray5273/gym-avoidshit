import gym
import gym_dodge
import random
env = gym.make('Dodge-v0')


for i in range(1000):
    done = False
    env.reset()
    env.render()
    while not done:
        action = env.action_space.sample()
        state,reward,done,_ = env.step(action)
        env.render()

