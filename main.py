import numpy as np
import torch as th
from environment import *
import os
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from neural_net import *

max_epi = 100000

test = False # True or false

action_dim = 2 # steering, acceleration
action_range = [[0.3*9.8,-9.8],[np.pi/40.,-np.pi/40.]] # action range [max_accel, max_decel], [max_angle_right, max_angle_left]
state_range = [[16.67,0.], [np.pi/4.,-np.pi/4.]] # state range  [max_vel, min_vel], [max_angle, min_angle]

env_size = np.array([ 1000., 20. ])
grid_range = np.array([ 100., 10. ])
cell_size = np.array([5,1])

max_vehicle = 20
safety_radius = 1

velo_range = numpy.array([10.,60.])/3.6
del_t = 0.1


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
print 'aa'

if cur_dir_chk.is_dir() is not True:
    os.mkdir(cur_dir+net_fol)

# import or generate network
net_chk = Path(net_fol + '/policy.pickle')
if net_chk.is_file() is True:
    with open(net_fol+'/policy.pickle') as net_restore:
        nets = pickle.load(net_restore)

else:
	a = 2    
#nets = net_module(network_type, network_spec)

### simulator
print 'a'

ego_state = numpy.array([0.,0.,0.,0.])
#obstacles = gen_obs(num_obs)
obstacles = [numpy.array([70.5,2.5,0.,0.]),numpy.array([45.2,-2.5,0.,0.]),numpy.array([37.9,0.1,0.,0.])]
done = 0
bump = 0
cnt = 0

# plt.show()

fig= plt.figure()
plt.ion()
while bump == 0 and done == 0:


    cnt += 1
    veh_grid = vehicle_input(ego_state, grid_range, obstacles, cell_size, env_size)
    # plt.ion()

    # plt.figure()
    plt.matshow(veh_grid[0])


    # plt.show()
    # plt.close()
    # plt.plot(ego_state)
    # fig.canvas.draw()


    # plt.close()

    ##################### action = neuralnet(veh_grid)
    ##################### policy needed
    action = numpy.array([10., 0.0])
    ego_state = step(ego_state, action, del_t, state_range)

    ## obstacle vehicles' action
    out_idx = []
    for obsidx in range(len(obstacles)):
        obs_action = surveh_model(obstacles[obsidx])
        obstacles[obsidx] = step(obstacles[obsidx],obs_action, del_t, state_range)
        if obstacles[obsidx][0] > env_size[0] or numpy.abs(obstacles[obsidx][1]) > env_size[1]/2:
            out_idx.append(obstacles[obsidx])

    for tmp in range(len(out_idx)):
        obstacles.remove(out_idx[tmp])

    done = chk_done(ego_state, obstacles, safety_radius, env_size)

    ############ reward and else
    ## [reward, done] = env_reward(ego_state, action, obstacles)
    if cnt is 100:
        done =1
