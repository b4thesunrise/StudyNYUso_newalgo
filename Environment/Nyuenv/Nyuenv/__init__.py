from gym.envs.registration import register

register(
    id='timevolume-v0',
    entry_point='Nyuenv.envs:Basicenv',
)