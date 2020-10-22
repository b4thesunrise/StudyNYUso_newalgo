import gym
import spinup
from spinup import sac_pytorch as sac
from spinup import ddpg_pytorch as ddpg
sac(env_fn=lambda : gym.make('Nyuenv:timevolume-v0'), max_ep_len = 365 * 100 + 1, lr = 5e-4)