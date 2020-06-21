import gym
import gym_avoidshit
import pybullet_envs
import numpy as np
from finalmodel import FinalModel

def main():
    env = gym.make('AvoidShit-v0')
    #env.render()
    model = FinalModel(env)
    avg = 0
	
    for i in range(100):
        score = playgame(model, env)
        avg += score
        print(f"run {i+1} total reward: {score}")
    print(avg/100)
def playgame(model, env):
    s = env.reset()
    reward_sum = 0
    while True:
        env.render()
        action = model.get_action(s)
        next_s, reward, done, _ = env.step(np.clip(action, env.action_space.low, env.action_space.high))
        s = next_s
        reward_sum += reward
        if done:
            break
    return reward_sum
    
if __name__ == "__main__":
    main()
