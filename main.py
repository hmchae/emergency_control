import numpy as np

import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from environment import *
from DDPG import *
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
steering_range = [[16.67,0.], np.pi+0.1] # state range  [max_vel, min_vel], [max_angle, min_angle]

env_size = np.array([ 400.,30. ])
grid_range = np.array([ 100., 10. ])
cell_size = np.array([5,1])

# max_vehicle = 50
safety_radius = 0.5


del_t = 0.05
num_obs = 5
num_lane = 5
#lane_dir = 2*(numpy.round(numpy.random.rand(num_lane))- 0.5)
lane_dir = -numpy.ones(num_lane)


## network initialization
rmcap = 10000
replay_memory = ReplayMemory(rmcap)



# set path
cur_dir = os.getcwd()
net_fol = '/networks'
cur_dir_chk = Path(cur_dir+net_fol)

if cur_dir_chk.is_dir() is not True:
    os.mkdir(cur_dir+net_fol)

# import or generate network
net_chk = Path(net_fol + '/policy.pickle')
if net_chk.is_file() is True:
    with open(net_fol+'/policy.pickle') as net_restore:
        nets = pickle.load(net_restore)

else:
    dummy = 1
    # nets = net_module(network_type, network_spec)



'''

vis_switch = 0
vis_FPS =  int(1/del_t)

ratio = env_size[1]/env_size[0]
scr_height = 1024
if vis_switch == 1:
    # screen = pygame.display.set_mode((scr_width,int(scr_width*ratio)))
    screen = pygame.display.init()

    screen = pygame.display.set_mode((int(scr_height*ratio),scr_height))
    background = pygame.transform.scale(pygame.image.load((cur_dir+'/images/background.png')),(int(scr_height*ratio),scr_height))
    for lane_draw in range(num_lane)[0:num_lane-1]:
        pygame.draw.line(background, (0,0,0), ((float(lane_draw)+1.)/float(num_lane)*ratio * scr_height,0),((float(lane_draw)+1.)/float(num_lane)*ratio * scr_height,scr_height), 5)
    screen.blit(background, (0,0))

    clock = pygame.time.Clock()
    clock.tick(vis_FPS)
    ego_veh_spr = CarSprite(cur_dir+'/images/car.png',(int((ego_state[1] + env_size[1] / 2) / env_size[1] * ratio * scr_height), int((env_size[0] - ego_state[0]) / env_size[0] * scr_height)),scr_height/env_size[0])
    ego_veh_group = pygame.sprite.RenderPlain(ego_veh_spr)
    obs_group_list = []


    for obsidx in range(len(obstacles)):
        obs_veh_spr = CarSprite(cur_dir+'/images/sur.png',(int((obstacles[obsidx][1] + env_size[1] / 2) / env_size[1] * ratio * scr_height), int((env_size[0] - obstacles[obsidx][0]) / env_size[0] * scr_height)),scr_height/env_size[0])
        obs_grp = pygame.sprite.RenderPlain(obs_veh_spr)
        obs_group_list.append(obs_grp)

sprite_group = pygame.sprite.RenderPlain()   #need to be checked

'''

# episode starts
ego_state = numpy.array([0.1,0.1,0.,0.])
action = numpy.array([0., 0.])
reward = 0.
obstacles,lane_vec = init_obs(num_obs,env_size,num_lane,lane_dir)
done = 0
bump = 0
cur_step = 0
veh_grid = numpy.zeros(gridmap.astype(int).tolist())
while bump == 0 and done == 0 :
    cur_step += 1

    #sprite_group.empty()

    if len(obstacles) < num_obs and random.random() < 1.:


        lane_idx = random.randrange(0,num_lane)
        dir = lane_dir[lane_idx]
        lane_vec.append(lane_idx)
        obs = numpy.array(
            [env_size[0] - (1. + dir) / 2 * env_size[0], ((1. + 2 * lane_idx) / (2 * num_lane) - 0.5) * env_size[1],
             random.random() * 10., (1. - dir) / 2 * numpy.pi])
        #print(obs)
        # if random.random() > 0.5:
        #     # obs = numpy.array([0.1,(random.random()-0.5)*env_size[1],random.random()*3.,0.])
        #     obs = numpy.array([0.1, ((1.+2*lane_idx)/(2*num_lane) - 0.5) * env_size[1], random.random() * 3., (1.-dir)/2*numpy.pi])
        # else:
        #     # obs = numpy.array([env_size[0]-0.1,(random.random()-0.5)*env_size[1],random.random()*3.,numpy.pi])
        #     obs = numpy.array(
        #         [env_size[0] - 0.1, ((1.+2*lane_idx)/(2*num_lane) - 0.5) * env_size[1], random.random() * 3., (1.-dir)/2*numpy.pi])
        obstacles.append(obs)
    '''
        if vis_switch is 1:
            obs_veh_spr = CarSprite(cur_dir + '/images/sur.png', (int((obstacles[-1][1] + env_size[1] / 2) / env_size[1] * ratio * scr_height), int((env_size[0] - obstacles[-1][0]) / env_size[0] * scr_height)),scr_height/env_size[0])
            obs_grp = pygame.sprite.RenderPlain(obs_veh_spr)
            obs_group_list.append(obs_grp)
            # obs_group_list.append(obs_veh_spr)
    '''

    prev_state = veh_grid
    veh_grid = vehicle_input(ego_state, grid_range, obstacles, cell_size, env_size) # gonna be input to the network(state)
    state = veh_grid


    '''
    if vis_switch == 1:



        ego_veh_group.update((int((ego_state[1] + env_size[1] / 2) / env_size[1] * ratio * scr_height),
                              int((env_size[0] - ego_state[0]) / env_size[0] * scr_height)),ego_state[-1])

        sprite_group.add(ego_veh_group)

        for obsidx in range(len(obstacles)):
            obs_grp = obs_group_list[obsidx]


            obs_grp.update((int((obstacles[obsidx][1] + env_size[1] / 2) / env_size[1] *ratio* scr_height),
                            int((env_size[0] - obstacles[obsidx][0]) / env_size[0] * scr_height)),obstacles[obsidx][-1])

            sprite_group.add(obs_grp)

        sprite_group.clear(screen,background)
        sprite_group.draw(screen)
        pygame.display.flip()
'''

    ##################### action = RL(veh_grid)
    ##################### policy needed
    replay_memory.push(prev_state, action, state, reward)

    action = numpy.array([0.01, 0.])

    ############ reward and else
    ## [reward, done] = env_reward(ego_state, action, obstacles)
    reward = 0



    ##DDPG learning

    [done, bump] = chk_done(ego_state, obstacles, safety_radius, env_size)
    if done == 1:
        print("episode ended at step" + int(cur_step))
        break

    ## ego vehicle's state update
    ego_state = step(ego_state, action, del_t, state_range,1)
    ## obstacle vehicles' action
    out_idx = []
    obs_action, obstacles = surveh_model(obstacles,lane_vec,action[0])
    for obsidx in range(len(obstacles)):
        obstacles[obsidx] = step(obstacles[obsidx],obs_action[obsidx], del_t, state_range,0)

        if obstacles[obsidx][1] == 0.:
            print(obstacles[obsidx])

        if obstacles[obsidx][0] > env_size[0] or numpy.abs(obstacles[obsidx][1]) > env_size[1]/2 or obstacles[obsidx][0] < 0 :
            out_idx.append(obsidx)

    reversed_idx = out_idx[::-1]

    for tmp in range(len(out_idx)):
        idx = reversed_idx[tmp]
        obstacles.pop(idx)
        lane_vec.pop(idx)
        '''
        if vis_switch is 1:
            obs_grp = obs_group_list[idx]
            obs_grp.clear(screen, background)
            obs_grp.empty()
            obs_group_list.pop(idx)
        '''






    # if cnt is 1000:
    #     done =1
