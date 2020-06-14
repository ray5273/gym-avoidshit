import gym
import gym_dodge
import gym_avoidshit
import random
env = gym.make('Dodge-v0')
#env = gym.make('AvoidShit-v0')

def get_action():
    return random.uniform(0,1)


for i in range(1000):
    done = False
    env.reset()
    env.render()
    while not done:
        action = get_action()
        state,reward,done,_ = env.step(action)
        env.render()

