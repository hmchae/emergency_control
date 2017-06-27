import numpy as np
import torch as th
from environment import *
import os
import sys
import pygame
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
from neural_net import *
from pygame.locals import *

max_epi = 100000

test = False # True or false

action_dim = 2 # steering, acceleration
action_range = [[0.3*9.8,-9.8],[np.pi/40.,-np.pi/40.]] # action range [max_accel, max_decel], [max_angle_right, max_angle_left]
state_range = [[16.67,0.], [np.pi/3.]] # state range  [max_vel, min_vel], [max_angle, min_angle]

env_size = np.array([ 100., 20. ])
grid_range = np.array([ 100., 10. ])
cell_size = np.array([5,1])

max_vehicle = 20
safety_radius = 0

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
ego_state = numpy.array([0.,0.,0.,0.])
#obstacles = gen_obs(num_obs)
obstacles = [numpy.array([70.5,2.5,0.,0.]),numpy.array([45.2,-2.5,0.,0.]),numpy.array([37.9,4.1,0.,0.])]

done = 0
bump = 0
cnt = 0

vis_switch = 1
vis_FPS =  int(1/del_t)

ratio = env_size[0]/env_size[1]
scr_width = 250

if vis_switch == 1:
    # screen = pygame.display.set_mode((int(env_size[1]),int(env_size[0])))
    screen = pygame.display.set_mode((scr_width,int(scr_width*ratio)))
    clock = pygame.time.Clock()
    clock.tick(vis_FPS)
    ego_veh_spr = CarSprite(cur_dir+'/images/car.png',int(env_size[0] - ego_state[0]))
    ego_veh_group = pygame.sprite.RenderPlain(ego_veh_spr)
    obs_group_list = []
    for obsidx in range(len(obstacles)):
        obs_veh_spr = CarSprite(cur_dir+'/images/car.png',int(env_size[0] - obstacles[obsidx][0]))
        obs_grp = pygame.sprite.RenderPlain(obs_veh_spr)
        obs_group_list.append(obs_grp)


while bump == 0 and done == 0:

    # print 'aaa'
    # print ego_state[0]
    cnt += 1
    veh_grid = vehicle_input(ego_state, grid_range, obstacles, cell_size, env_size)
    # pdb.set_trace()

    if vis_switch == 1:
        ego_veh_group.update((int((ego_state[1]+env_size[1]/2)/env_size[1]*scr_width),int((env_size[0] - ego_state[0])/env_size[0]*ratio*scr_width)))
        screen.fill((128, 128, 128))
        ego_veh_group.draw(screen)
        for obsidx in range(len(obstacles)):
            obs_grp = obs_group_list[obsidx]
            obs_grp.update((int((obstacles[obsidx][1]+env_size[1]/2)/env_size[1]*scr_width),int((env_size[0] - obstacles[obsidx][0])/env_size[0]*ratio*scr_width)))
            obs_grp.draw(screen)


        pygame.display.flip()


    ##################### action = neuralnet(veh_grid)
    ##################### policy needed
    action = numpy.array([0.01, 0.0])
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
    # if cnt is 1000:
    #     done =1
