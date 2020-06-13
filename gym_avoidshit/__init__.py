from gym.envs.registration import register

register(
        id='AvoidShit-v0',
        entry_point='gym_avoidshit.envs:AvoidShitEnv',
)
