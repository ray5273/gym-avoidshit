import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense

class _Actor_Build(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(_Actor_Build, self).__init__()
        self.action_size = action_size
        self.layer1 = Dense(512, activation="relu")
        self.layer2 = Dense(256, activation="relu")
        self.layer_mean = Dense(action_size)
        self.layer_std = Dense(action_size)
        dummy_state = tf.constant(np.zeros(shape=(1,)+state_size, dtype=np.float32))
        self(dummy_state)

    def get_state_out(self, state):
        state_out = self.layer1(state)
        state_out = self.layer2(state_out)
        return state_out

    def get_mean(self, state):
        mean = self.layer_mean(state)
        #mean *= mean
        return mean

    def get_std(self, state):
        std = self.layer_std(state)
        #std += 1e-10
        return std

    def get_dist(self, state):
        state = self.get_state_out(state)
        mean = self.get_mean(state)
        std = self.get_std(state)
        std = tf.clip_by_value(std, -2, 2)
        return mean, std

    def sample(self, mean, std):
        return mean + tf.random.normal(shape=mean.shape)*tf.math.exp(std)

    def log_prob(self, action, mean, std):
        normal_mu = (action - mean) / tf.exp(std)
        std_ = -tf.reduce_sum(std, axis=-1)
        normal_ = -0.5*tf.reduce_sum(tf.square(normal_mu), axis=-1)
        constant_ = -0.5*self.action_size*tf.math.log(2*np.pi)
        return std_ + normal_ + constant_

    def call(self, state):
        mean, std = self.get_dist(state)
        action = self.sample(mean, std)
        log_prob = self.log_prob(action, mean, std)
        action = tf.tanh(mean)
        return action*2., log_prob, std

class FinalModel:
    def __init__(self, env):
        state_size = env.observation_space.shape
        action_size = env.action_space.high.size
        self.actor = _Actor_Build(state_size, action_size)
        #self.actor.load_weights("./fix_yes_normal_200000steps/action.h5")
        #self.actor.load_weights("./random_yes_normal_1000000steps/action.h5")
        #self.actor.load_weights("./random_yes_normal_2000000steps/action.h5")
        self.actor.load_weights("./random_yes_normal_3000000steps/action.h5")
    def get_action(self, state):
        state = np.expand_dims(state, axis=0).astype(np.float32)
        action, _, _ = self.actor(tf.constant(state))
        return action.numpy()[0]
