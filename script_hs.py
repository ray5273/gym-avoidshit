import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import gym
import gym_avoidshit
from dqn_hs import DQN

os.environ["CUDA_VISIBLE_DEVICES"]='-1'

orgDQN, multistep = False, False
if len(sys.argv) == 1:
    print("There is no argument, please input")

for i in range(1,len(sys.argv)):
    if sys.argv[i] == "orgDQN":
        orgDQN = True
    elif sys.argv[i] == "multistep":
        multistep = True


env = gym.make('AvoidShit-v0')

if orgDQN:
    env.reset()
    dqn = DQN(env, multistep=False)
    orgDQN_record = dqn.learn(30000)
    del dqn

if multistep:
    env.reset()
    dqn = DQN(env, multistep=True)
    multistep_record = dqn.learn(30000)
    del dqn

print("Reinforcement Learning Finish")
print("Draw graph ... ")

if orgDQN:
    plt.plot(np.arange((len(orgDQN_record))), orgDQN_record, label='Orginal DQN')
if multistep:
    plt.plot(np.arange((len(multistep_record))), multistep_record, label='Multistep DQN')

plt.legend()
fig =plt.gcf()
plt.savefig("hs_dqn_again_result.png")
plt.show()
