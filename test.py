import sys
import numpy as np
import tensorflow as tf
import random
import gym
import gym_avoidshit
from collections import deque
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

env = gym.make('AvoidShit-v0')

def get_action():
    return random.randint(-1,1)

def loadModel():
    model_path='models/hs_dqn29900.h5'
    restored_model = tf.keras.models.load_model(model_path)
    return restored_model

def predict(state, localNet):
    target = localNet.predict_on_batch(state)
    action = np.argmax(target)
    return action

model = loadModel()
for i in range(100):
    done = False
    state = env.reset()
    while not done:
        action = predict(state,model)
        state,reward,done,score = env.step(action)
        env.render()
    print("episode ",i,"  score : ",score)
