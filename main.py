import numpy as np
import torch as th
from environment import *
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from  neural_net import *

max_epi = 100000

test = False # True or false

action_dim = 2 # steering, acceleration
action_range = [[0.3*9.8,-9.8],[np.pi/40,-np.pi/40]] # action range [max_accel, max_decel], [max_angle_right, max_angle_left]
state_range = [[16.67,0], [np.pi/4,-np.pi/4]] # state range  [max_vel, min_vel], [max_angle, min_angle]

env_size = np.array([ 1000, 20 ])
grid_range = np.array([ 100, 10 ])
cell_size =  np.array([5,1])



network_type = ['conv','conv','conv','linear']
network_spec = [ [1,30,3,'ReLU'],
                 [30,20,3,'ReLU'],
                 [20,10,2,'ReLU'],
                 [500, 'ReLU'],
                 [100,'ReLU'],
                 [action_dim,'None']]



## network initialization
# set path
cur_dir = os.getcwd()
net_fol = 'networks'
cur_dir_chk = Path(cur_dir+net_fol)

if cur_dir_chk.is_dir() is not True:
    os.mkdir(cur_dir+net_fol)

# import or generate network
net_chk = Path(net_fol + '/policy.pickle')
if net_chk.is_file() is True:
    with open(net_fol+'/policy.pickle') as net_restore:
        nets = pickle.load(net_restore)

else:
    nets = net_module(network_type, network_spec)

##
for epi_idx in range(max_epi):