import gym
import gym_avoidshit
import random
env = gym.make('AvoidShit-v0')

def get_action():
	return random.randint(-1,1)

def get_continuous_action():
	return random.uniform(-1.0, 1.0)

for i in range(1000):
	done = False
	env.reset()
	env.render()
	while not done:
		#action = get_action()
		action = get_continuous_action()
		print("Action : ", action)
		state,reward,done,_ = env.step(action)
		env.render()
