import sys
import numpy as np
#import tensorflow.compat.v1 as tf
import tensorflow as tf
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape, Dropout
from tensorflow.keras.optimizers import Adam
import random
import gym
from collections import deque
DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 100000
LEARNING_RATE = 0.001
LEARNING_START = 2000
BATCH_SIZE = 32
STEP_SIZE = 2

class DQN:
        def __init__(self, env, multistep=False):
                REPLAY_MEMORY=100000
                if(multistep == False):
                        self.buf_size = REPLAY_MEMORY
                        self.update_period = 2
                        LEARNING_RATE = 0.001
                        DISKCOUNT_RATE = 0.99
                        self.tow = 1
                else:
                        self.buf_size = REPLAY_MEMORY
                        LEARNING_RATE = 0.001
                        self.update_period = 2
                        self.tow = 1

                self.env = env
                #self.state_size = self.env.observation_space.shape[0]
                self.state_size = 22
                #self.action_size = self.env.action_space.n
                self.action_size = 3

                self.multistep = multistep
                self.n_steps = STEP_SIZE
                self.step_buf = deque(maxlen=self.buf_size)
                self.state_buf = deque(maxlen=self.n_steps)
                self.nstate_buf = deque(maxlen=self.n_steps)
                self.action_buf = deque(maxlen=self.n_steps)
                self.reward_buf = deque(maxlen=self.n_steps)
                self.done_buf = deque(maxlen=self.n_steps)

                self.buf = deque(maxlen=REPLAY_MEMORY)
                self.epsilon = 0.0
                self.optimize = Adam(LEARNING_RATE)

                self.net = self._build_network()
                self.target = self._build_network()
                self.copy_weight()


        #def store2(self, lst):
        #       for i in range(len(lst)):
        #               state = lst[i][0]
        #               action = lst[i][1]
        #               nstate = lst[i][2]
        #               reward = lst[i][3]
        #               done = lst[i][4]
        #               self.store(state, action, nstate, reward, done)

        def save(self,save_path):
            self.net.save(save_path)

        def store(self, state, action, nstate, reward, done, step_count):
                index = random.randint(0, 2)
                if(index % 2 == 0):
                        self.buf.append((state, action, nstate, reward, done))
                else:
                        self.buf.appendleft((state, action, nstate, reward, done))
                if(self.multistep == True and self.n_steps != 1):
                        self.state_buf.append(state)
                        self.action_buf.append(action)
                        self.nstate_buf.append(nstate)
                        self.reward_buf.append(reward)
                        self.done_buf.append(done)
                        if((step_count+1) >= self.n_steps):
                                sum_reward = 0
                                for i in range(self.n_steps):
                                        state = self.state_buf.popleft()
                                        action = self.action_buf.popleft()
                                        nstate = self.nstate_buf.popleft()
                                        reward = self.reward_buf.popleft()
                                        done = self.done_buf.popleft()
                                        if(i == 0):
                                                init_state = np.copy(state)
                                                init_action = np.copy(action)
                                        else:
                                                self.state_buf.append(state)
                                                self.action_buf.append(action)
                                                self.nstate_buf.append(nstate)
                                                self.reward_buf.append(reward)
                                                self.done_buf.append(done)
                                        sum_reward += reward*np.power(DISCOUNT_RATE, i)
                                if(index % 2 == 0):
                                        self.step_buf.append((init_state, init_action, nstate, sum_reward, done))
                                else:
                                        self.step_buf.appendleft((init_state, init_action, nstate, sum_reward, done))
                                #self.step_buf.append((init_state, init_action, nstate, sum_reward, done))

        def copy_weight(self, ):
                self.target.set_weights(self.net.get_weights())

        def _build_network(self, ):
                model = Sequential()
                model.add(Dense(512, input_dim=self.state_size, activation="relu"))
                model.add(Dense(256, activation="relu"))
                model.add(Dense(self.action_size))
                model.compile(loss='mse', optimizer=self.optimize)
                return model

        #def optimizer(self):
        #       one_hot = tf.one_hot(self.action_input, self.y_input)
        #       Q_action = tf.reduce_sum(tf.muliply(self.net, self.action_input), axis=1)
        #       self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        #       self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.cost)

        def predict(self, state):
                if(np.random.random_sample() < self.epsilon):
                        return self.env.action_space.sample()
                else:
                        value = self.net.predict_on_batch(state)
                        return np.argmax(value[0])

        def train_minibatch(self, ):
                if(self.multistep == True and self.n_steps != 1):
                        minibatch = random.sample(self.step_buf, BATCH_SIZE)
                else:
                        minibatch = random.sample(self.buf, BATCH_SIZE)

                #action_batch = np.array([data[1] for data in minibatch])
                #reward_batch = np.array([data[3] for data in minibatch])
                #done_batch = np.array([data[4] for data in minibatch])
                state_batch = np.array([data[0] for data in minibatch])
                action_batch = [data[1] for data in minibatch]
                nstate_batch = np.array([data[2] for data in minibatch])
                reward_batch = [data[3] for data in minibatch]
                done_batch = [data[4] for data in minibatch]

                state_batch = np.reshape(state_batch, (BATCH_SIZE, -1))
                nstate_batch = np.reshape(nstate_batch, (BATCH_SIZE, -1))

                out = self.net.predict_on_batch(state_batch) ## Q-value
                target = np.copy(out)
                qv = self.target.predict_on_batch(nstate_batch) ## Target Q-value
                max_action_batch = self.net.predict_on_batch(nstate_batch)
                for i in range(BATCH_SIZE):
                        max_action = np.argmax(max_action_batch[i])
                        max_q = qv[i][max_action]
                        #max_q = max(qv[i])
                        target[i][action_batch[i]] = (reward_batch[i] + \
                                (DISCOUNT_RATE*max_q if not done_batch[i] else 0))

                #loss = self.net.train_on_batch(state_batch, out)
                #self.net.compile(loss='mse', optimizer=self.optimize)
                loss = self.net.train_on_batch(state_batch, target)
                return loss

        def update_epsilon(self, episode, avg):
                #if(avg > 20):
                self.epsilon = 1.0 / ((episode // 1) + 1)
                #else:
                #       self.epsilon = 1.0 / ((avg // 1) + 1)
                pass


        # episode 최대 횟수는 구현하는 동안 더 적게, 더 많이 돌려보아도 무방합니다.
        # 그러나 평가시에는 episode 최대 회수를 1500 으로 설정합니다.
        def learn(self, max_episode=1500):
                avg_step_count_list = []     # 결과 그래프 그리기 위해 script.py 로 반환
                last_100_episode_step_count = deque(maxlen=100)
                #one_step_before = 0

                for episode in range(max_episode):
                        done = False
                        state = self.env.reset()
                        step_count = 0
                        avg_step_count = 0
                        lst = []
                        while not done:
                                state = np.reshape(state, (1, -1))
                                action = self.predict(state)
                                next_state, reward, done, _ = self.env.step(action)
                                self.store(state, action, next_state, reward, done, step_count)
                                #lst.append((state, action, next_state, reward, done, step_count))
                                state = next_state
                                step_count += 1
                        #dif = step_count - one_step_before
                        #one_step_before = step_count
                        #if(dif >= 0):
                        #       print(dif, end = "")
                        #       self.store2(lst)
                        if(step_count > 400):
                                self.optimize = Adam(LEARNING_RATE*0.1)
                        else:
                                self.optimize = Adam(LEARNING_RATE)

                        if(episode % self.update_period == 0):
                                self.copy_weight()
                        losses = 0
                        if(len(self.buf) > LEARNING_START):
                                for i in range(50):
                                        losses = self.train_minibatch()
                        last_100_episode_step_count.append(step_count)

                        #if len(last_100_episode_step_count) == 100:
                        avg_step_count = np.mean(last_100_episode_step_count)
                        print("[Episode {:>5}]  episode step_count: {:>5} episode score : {} avg step_count: {}[{}]".format(episode, step_count,_, avg_step_count, losses))
                        # if(avg_step_count >= 475.0):
                        #         break
                        avg_step_count_list.append(avg_step_count)
                        self.update_epsilon(episode, avg_step_count)
                        if episode%100 == 0 and episode!=0:
                            save_path = 'models/'
                            if self.multistep:
                                save_path += "hs_multidqn10akstwostep"+str(episode)+'.h5'
                            else:
                                save_path += "hs_dqn"+str(episode)+'.h5'
                            self.save(save_path)
                return avg_step_count_list