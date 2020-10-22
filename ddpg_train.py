import gym
import spinup
from spinup import ddpg_pytorch as ddpg
ddpg(env_fn=lambda : gym.make('Nyuenv:timevolume-v0'), max_ep_len = 365 * 100 + 1, pi_lr = 5e-4, q_lr = 1e-3)