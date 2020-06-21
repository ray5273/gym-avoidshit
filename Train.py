import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pybullet_envs, logging, gym
import gym_avoidshit
from SAC import Agent
from cpprb import ReplayBuffer
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque

class Trainer:
	def __init__(self, env, policy):
		self._policy = policy
		self._env = env
		self._max_steps = 3000000
		self.n_warmup = 10000
		self._episode_max_steps = 1000
		self.update_interval = 1

	def __call__(self):
		total_steps = 0
		episode_steps = 0
		episode_return = 0
		n_episode = 0
		#logging.basicConfig(filename='./co_model.log', level=logging.DEBUG)
		replay_buffer = ReplayBuffer(2000000, 
							env_dict={"obs": {"shape": env.observation_space.shape},
									"act": {"shape": env.action_space.shape},
									"next_obs": {"shape": env.observation_space.shape},
									"rew" : {},
									"done" : {}})
		obs = self._env.reset()
		last_100_episode = deque(maxlen=100)
		avg_return = []
		while total_steps < self._max_steps:
			if total_steps < self.n_warmup:
				action = self._env.action_space.sample()
			else:
				action = self._policy.get_action(obs)

			clip_action = np.clip(action, env.action_space.low+1e-10, env.action_space.high-1e-10)
			next_obs, reward, done, _ = self._env.step(clip_action)
			episode_steps += 1
			episode_return += reward
			total_steps += 1
			#env.render()
			done_flag = done
			if hasattr(self._env, "_max_episode_steps") and \
					episode_steps == self._env._max_episode_steps:
				done_flag = False
			replay_buffer.add(obs=obs, act=action, next_obs=next_obs, rew=reward, done=done_flag)
			obs = next_obs

			if done or episode_steps == self._episode_max_steps:
				obs = self._env.reset()

				n_episode += 1
				print("Epi : {}, Steps : {}/{}, Reward : {}".format\
						(n_episode, episode_steps, total_steps, episode_return))
				last_100_episode.append(episode_return)
				avg_return.append(np.mean(last_100_episode))
				#logging.debug("Epi : {}, Steps : {}/{}, Reward : {}".format\
				#		(n_episode, episode_steps, total_steps, episode_return))
				episode_steps = 0
				episode_return = 0
			if total_steps < self.n_warmup:
				continue

			if total_steps % self.update_interval == 0:
				samples = replay_buffer.sample(256)
				td_error = self._policy.train(samples["obs"], samples["act"], samples["next_obs"],samples["rew"], np.array(samples["done"], dtype=np.float32), None)
			
		now = datetime.now()
		name = str(now.day)+"_"+str(now.hour)+"_"+str(now.minute)+"_"+\
			str(now.second)+str(total_steps)
		self._policy.actor.save_weights(name+"random3.h5")
		self._policy.q1.save_weights(name+"q13.h5")
		self._policy.q2.save_weights(name+"q23.h5")
		self._policy.origin_v.save_weights(name+"origin_v3.h5")
		self._policy.target_v.save_weights(name+"target_v3.h5")
		plt.plot(np.arange((len(avg_return))), avg_return, label='Return')
		plt.legend()
		fig = plt.gcf()
		plt.savefig("result_random_normalization3.png")
		plt.show()
if __name__ == '__main__':
	#env = gym.make("AntBulletEnv-v0")
	#env = gym.make("Pendulum-v0")
	#env = gym.make("MountainCarContinuous-v0")
	env = gym.make('AvoidShit-v0')
	state_size = env.observation_space.shape
	action_size = env.action_space.high.size
	action_high = env.action_space.high
	action_low = env.action_space.low
	agent = Agent(state_size, action_size, action_high, action_low, True)
	train = Trainer(env, agent)
	train()
