import gym
env = gym.make('Nyuenv:timevolume-v0')
env.reset()
for i in range(366 * 24):
    a,b,c,d = env.step(-0.1)
    if i % 24 == 0:
        print(i/24, b,c,d)
        print(a.shape)