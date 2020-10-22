import math
import numpy as np
import pandas as pd
import gym
from scipy.stats import describe
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt
import copy
from random import randint
# plan to add setting: 
## save energy
## bid on different region, need different constraint and different combination, constraint volume
# plan to solve the hear to learn problem
## use less info
## use simpler algo like dqn
## use simple settings like classification
class Basicenv(gym.Env):
    def __init__(self, locs = [0,1,2,3,4,20,21,22,23,24,25,26,27,28,29,44,45,46,47,48,49,50,51,52,53], start_budget = 10000.0 , volume = 30, da_path = '/home/pku1616/cdf/StudyNYUso_newalgo/Database/adjust_timelabel_dadf.csv', rt_path = '/home/pku1616/cdf/StudyNYUso_newalgo/Database/adjust_timelabel_rtdf.csv', std_array_path = '/home/pku1616/cdf/StudyNYUso_newalgo/Database/LONGIL_trading_risk.npy', seed = None, numberlimit = 2, statelen = 25, location = 'LONGIL', beta = 0.01):
        #data setting
        self.dadata = pd.read_csv(da_path, index_col = 0).drop(columns = ['Month','Hour','Quarter','day'])[location]
        self.dadata.index = pd.to_datetime(self.dadata.index)
        self.rtdata = pd.read_csv(rt_path, index_col = 0).drop(columns = ['Month','Hour','Quarter','day'])[location]
        self.rtdata.index = pd.to_datetime(self.rtdata.index)
        self.timearray = np.load(std_array_path)
        self.datapointer = randint(10, 7200)#从第四天开始
        self.epoch = 0
        #budget setting
        self.budget = 0
        self.budget_const = start_budget
        self.numberlimit = numberlimit
        #volume setting
        self.volume_const = volume
        #set the reinforcement setting
        self.action_space = spaces.Box(low = -1, high = 1, shape = (1,), dtype=np.float32)
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (statelen*2 + 4,),dtype=np.float32)
        self.seednumber = self.seed(seed)
        self.beta = beta
        #tracking
        self.actions = []
        self.gains = []
   
    def construct_state(self):
        loc1 = self.datapointer * 24 + self.hour
        loc2 = (self.datapointer - 1)*24 + self.hour
        loc3 = (self.datapointer - 2)*24 + self.hour
        locindex = self.dadata.index[loc1]
        data = np.concatenate( (self.dadata.iloc[loc1-5:loc1].to_numpy(), self.dadata.iloc[loc2-5:loc2 + 5].to_numpy(), self.dadata.iloc[loc3-5:loc3 + 5].to_numpy(), self.rtdata.iloc[loc1-5:loc1].to_numpy(), self.rtdata.iloc[loc2-5:loc2 + 5].to_numpy(), self.rtdata.iloc[loc3-5:loc3 + 5].to_numpy(), np.array([locindex.month, locindex.day, locindex.hour, self.budget])), axis = 0)
        self.state = np.array(data, dtype=np.float32)
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        self.datapointer = randint(10, 7200)
        self.epoch = 0
        self.hour = 0
        self.budget = self.budget_const
        self.actions = []
        self.gains = []
        self.construct_state()
        return self.state
    
    def show(self):
        plt.plot(self.actions)
        plt.show()
        plt.plot(self.gains)
        plt.show()
    
    def updatetime(self, action):
        self.hour += 1
        if self.hour == 24:
            self.hour = 0
            self.datapointer += 1
            self.epoch += 1
            print(self.epoch, self.budget, action)
            print
        if self.epoch == self.volume_const:
            return True
        else:
            return False
        
    def cal_var(self):
        loc1 = self.datapointer * 24 + self.hour
        locindex = self.dadata.index[loc1]
        #print('var:',self.timearray[locindex.month-1][locindex.day-1][locindex.hour-1])
        return self.timearray[locindex.month-1][locindex.day-1][locindex.hour-1]
        
    
    def step(self, action):
        done = False
        gain = 0
        pda = self.dadata.iloc[int(self.datapointer * 24 + self.hour)]
        prt = self.rtdata.iloc[int(self.datapointer * 24 + self.hour)]
        #print(self.dadata.index[int(self.datapointer * 24 + self.hour)], self.rtdata.index[int(self.datapointer * 24 + self.hour)])
        volume = self.numberlimit * action
        self.actions.append(float(volume))
        if volume >= 0:#买入pda
            gain += volume * (prt - pda)
        else:
            gain += -volume * (pda - prt)
        #print('gain:', gain, 'volume:', volume, pda, prt)
        #先计算收益变化再计算一般变化
        self.budget += gain
        self.gains.append(float(copy.deepcopy(self.budget)))
        #先计算收益变化再计算风险变化
        gain -= self.cal_var() * volume * self.beta
        #print('gain:', gain)
        #print(self.budget)
        if self.budget <= 0:
            done = True
        self.construct_state()
        done = self.updatetime(action) or done
        self.show()
        #print(self.budget, self.epoch)
        if done:
            print(describe(self.actions))
            print(describe(self.gains))
            print('----------------------------------------------')
        return self.state, gain, done, {'bud': self.budget}

