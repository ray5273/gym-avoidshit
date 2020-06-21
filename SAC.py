import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gym
import pybullet_envs
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp

def copy(target, source, tau):
	def update_op(target, source, tau):
		return target.assign(tau*source + (1.0-tau)*target, False)
	update_ops = [update_op(target_var, source_var, tau) \
					for target_var, source_var in zip(target, source)]
	return tf.group(*update_ops)

def huber(x, delta=1.0):
	delta = tf.ones_like(x) * delta
	__min = 0.5 * tf.square(x)
	__max = delta * (tf.abs(x) - 0.5 * delta)
	return tf.where(tf.abs(x)<=delta, x=__min, y=__max)

class _Qnet_Build(tf.keras.Model):
	def __init__(self, state_size, action_size):
		super(_Qnet_Build, self).__init__()
		self.layer1 = Dense(512, activation="relu")
		self.layer2 = Dense(256, activation="relu")
		self.layer3 = Dense(1, activation="linear")
		dummy_state = tf.constant(np.zeros(shape=(1, state_size[0]), dtype=np.float32))
		dummy_action = tf.constant(np.zeros(shape=[1, action_size], dtype=np.float32))
		self([dummy_state, dummy_action])

	def call(self, inputs):
		[state, action] = inputs
		__inputs = tf.concat([state, action], axis=1)
		output = self.layer1(__inputs)
		output = self.layer2(output)
		output = self.layer3(output)
		return tf.squeeze(output, axis=1)

class _Vnet_Build(tf.keras.Model):
	def __init__(self, state_size, action_size):
		super(_Vnet_Build, self).__init__()
		self.layer1 = Dense(512, activation="relu")
		self.layer2 = Dense(256, activation="relu")
		self.layer3 = Dense(1, activation="linear")
		dummy_state = tf.constant(np.zeros(shape=(1, state_size[0]), dtype=np.float32))
		self(dummy_state)

	def call(self, inputs):
		output = self.layer1(inputs)
		output = self.layer2(output)
		output = self.layer3(output)
		return tf.squeeze(output, axis=1)

class _Actor_Build(tf.keras.Model):
	def __init__(self, state_size, action_size, ah):
		super(_Actor_Build, self).__init__()
		self.ah = ah
		self.action_size = action_size
		self.layer1 = Dense(512, activation="relu")
		self.layer2 = Dense(256, activation="relu")
		self.layer_mean = Dense(action_size)
		self.layer_std = Dense(action_size)
		dummy_state = tf.constant(np.zeros(shape=(1, state_size[0]), dtype=np.float32))
		self(dummy_state)

	def get_state_out(self, state):
		state_out = self.layer1(state)
		state_out = self.layer2(state_out)
		return state_out

	def get_mean(self, state):
		mean = self.layer_mean(state)
		return mean

	def get_std(self, state):
		std = self.layer_std(state)
		return std

	def get_dist(self, state):
		state = self.get_state_out(state)
		mean = self.get_mean(state)
		std = self.get_std(state)
		std = tf.clip_by_value(std, -20.0, 2.0)
		return mean, std

	def sample(self, mean, std):
		return mean + tf.random.normal(shape=mean.shape)*tf.math.exp(std)

	def log_prob(self, action, mean, std):
		normal_mu = (action - mean) / (tf.exp(std)+1e-6)
		std_ = -tf.reduce_sum(std, axis=-1)
		normal_ = -0.5*tf.reduce_sum(tf.square(normal_mu), axis=-1)
		constant_ = -0.5*self.action_size*tf.math.log(2*np.pi)
		return std_ + normal_ + constant_

	def correction(self, log_prob, action):
		diff = tf.reduce_sum(tf.math.log(1.0 - action**2 + 1e-6), axis=1)
		return log_prob - diff

	def call(self, state):
		mean, std = self.get_dist(state)
		action = self.sample(mean, std)
		log_prob = self.log_prob(action, mean, std)
		#print("what : ", action)
		action = tf.tanh(action)
		log_prob = self.correction(log_prob, action)
		#print("what : ", action)
		return action*self.ah, log_prob, std


class Agent:
	def __init__(self, state_size, action_size, ah, al, extension):
		self.Q_lr = self.V_lr = self.Actor_lr = 3e-4
		self.Qnet_Build(state_size, action_size, self.Q_lr)
		self.Vnet_Build(state_size, action_size, self.V_lr)
		self.Actor_Build(state_size, action_size, self.Actor_lr, ah)
		
		self.extension = extension
		if(extension):
			self.log_alpha = tf.Variable(0., dtype=tf.float32)
			self.alpha = tf.Variable(0., dtype=tf.float32)
			self.alpha.assign(tf.exp(self.log_alpha))
			self.target_alpha = -action_size
			self.alpha_optimizer = Adam(3e-4)
		else:
			self.alpha = 0.05

		self.discount = 0.98
	
	def Qnet_Build(self, state_size, action_size, lr):
		self.q1 = _Qnet_Build(state_size, action_size)
		self.q2 = _Qnet_Build(state_size, action_size)
		#self.q1.load_weights("./random_yes_normal_2000000steps/q1.h5")
		#self.q2.load_weights("./random_yes_normal_2000000steps/q2.h5")
		self.q1_optimizer = Adam(lr)
		self.q2_optimizer = Adam(lr)

	def Vnet_Build(self, state_size, action_size, lr):
		self.origin_v = _Vnet_Build(state_size, action_size)
		self.target_v = _Vnet_Build(state_size, action_size)
		#self.origin_v.load_weights("./random_yes_normal_2000000steps/origin_value.h5")
		#self.target_v.load_weights("./random_yes_normal_2000000steps/target_value.h5")
		copy(target=self.target_v.weights, source=self.origin_v.weights, tau=1.0)
		self.origin_v_optimizer = Adam(lr)

	def Actor_Build(self, state_size, action_size, lr, ah):
		self.actor = _Actor_Build(state_size, action_size, ah)
		#self.actor.load_weights("./random_yes_normal_2000000steps/action.h5")
		self.actor_optimizer = Adam(lr)

	def get_action(self, state):
		state = np.expand_dims(state, axis=0).astype(np.float32)
		action, _, _ = self.actor(tf.constant(state))
		return action.numpy()[0]

	@tf.function
	def __train(self, state, action, nstate, reward, done, weight):
		with tf.device("/cpu:1"):
			reward = tf.squeeze(reward, axis=1)
			not_done = 1. - tf.cast(done, dtype=tf.float32)
			with tf.GradientTape(persistent=True) as tape:
				cur_q1 = self.q1([state, action])
				cur_q2 = self.q2([state, action])
				next_target = self.target_v(nstate)
				target_q = tf.stop_gradient(reward+not_done*self.discount*next_target)

				td_loss_q1 = tf.reduce_mean(huber(target_q-cur_q1, delta=10.)*weight)
				td_loss_q2 = tf.reduce_mean(huber(target_q-cur_q2, delta=10.)*weight)
				origin_v = self.origin_v(state)
				act, logp, _ = self.actor(state)
				act = tf.cast(act, dtype=tf.float32)
				cur_q1 = self.q1([state, act])
				cur_q2 = self.q2([state, act])
				cur_min = tf.minimum(cur_q1, cur_q2)
				target_v = tf.stop_gradient(cur_min-self.alpha*logp)
				td_error = target_v - origin_v
				td_loss = tf.reduce_mean(huber(td_error, delta=10.)*weight)

				policy_loss = tf.reduce_mean((self.alpha*logp - cur_min)*weight)
				if(self.extension):
					alpha_ = tf.stop_gradient(logp + self.target_alpha)
					alpha_loss = -tf.reduce_mean(self.log_alpha * alpha_)
			q1_grad = tape.gradient(td_loss_q1, self.q1.trainable_variables)
			self.q1_optimizer.apply_gradients\
				(zip(q1_grad, self.q1.trainable_variables))
			q2_grad = tape.gradient(td_loss_q2, self.q2.trainable_variables)
			self.q2_optimizer.apply_gradients\
				(zip(q2_grad, self.q2.trainable_variables))
			
			v_grad = tape.gradient(td_loss, self.origin_v.trainable_variables)
			self.origin_v_optimizer.apply_gradients\
				(zip(v_grad, self.origin_v.trainable_variables))
			copy(target=self.target_v.weights, source=self.origin_v.weights, tau=0.01)
			
			actor_grad = tape.gradient(policy_loss, self.actor.trainable_variables)
			self.actor_optimizer.apply_gradients\
				(zip(actor_grad, self.actor.trainable_variables))
			
			if(self.extension):
				alpha_grad = tape.gradient(alpha_loss, [self.log_alpha])
				self.alpha_optimizer.apply_gradients\
					(zip(alpha_grad, [self.log_alpha]))
				grad = tf.exp(self.log_alpha)
				clip = tf.clip_by_value(\
					tf.where(tf.math.is_nan(grad), 0.0008, grad), 0.0008, 0.2)
				self.alpha.assign(clip)
			del tape
		return td_error

	def train(self, state, action, nstate, reward, done, weights=None):
		if weights is None:
			weights = np.ones_like(reward)
		td_error = self.__train(state, action, nstate, reward, done, weights)
		return td_error
