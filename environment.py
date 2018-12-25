import pygame
import numpy
import math
import pdb
import matplotlib.pyplot as plt
import random


def vehicle_input(ego_state, sensor_range, obstacles, cell_size, env_border):

    # ego_state/obstacles : [numpy.array([x,y,velocity,angle])]
    # sensor_range : numpy.array([longitudinal range, lateral range])

    gridmap = sensor_range/cell_size
    grid_out = numpy.zeros(gridmap.astype(int).tolist())
    lb_point = numpy.array([ego_state[0] + sensor_range[1]/2*math.sin(ego_state[3]),
                            ego_state[1] - sensor_range[1]/2*math.cos(ego_state[3])])

    rb_point = numpy.array([ego_state[0] - sensor_range[1]/2*math.sin(ego_state[3]),
                            ego_state[1] + sensor_range[1]/2*math.cos(ego_state[3])])

    lt_point = numpy.array([lb_point[0] + sensor_range[0]*math.cos(ego_state[3]),
                            lb_point[1] + sensor_range[0]*math.sin(ego_state[3])])

    rt_point = numpy.array([rb_point[0] + sensor_range[0] * math.cos(ego_state[3]),
                            rb_point[1] + sensor_range[0] * math.sin(ego_state[3])])

    mt_point = (lt_point + rt_point)/2


    for idx in range(len(obstacles)):
        obs_state = obstacles[idx]
        dif_vec = (obs_state[0:2]-ego_state[0:2])

        obj2obs = numpy.sqrt((dif_vec**2).sum())

        js_angle = numpy.arccos((((mt_point-ego_state[0:2]) * (obs_state[0:2]-ego_state[0:2])).sum()/numpy.sqrt(((mt_point-ego_state[0:2])**2).sum()*((obs_state[0:2]-ego_state[0:2])**2).sum())))


        if js_angle < numpy.pi/2 and obj2obs*math.sin(js_angle) < sensor_range[1]/2 and obj2obs*math.cos(js_angle) < sensor_range[0] :
            loc_x = obj2obs*math.cos(js_angle)
            loc_y = obj2obs*math.sin(js_angle)

            if numpy.linalg.norm(obs_state[0:2]-lb_point) > numpy.linalg.norm(obs_state[0:2]-rb_point):
                dir = 1
            else:
                dir = -1

            loc_y = loc_y*dir
            loc_tmp = numpy.floor(numpy.array([loc_x,loc_y])/numpy.array(cell_size)*numpy.array([-1.,1.])) + numpy.array([gridmap[0],numpy.ceil(gridmap[1]/2)])
            grid_loc = loc_tmp.astype(int).tolist()

            grid_out[grid_loc[0]][grid_loc[1]] = 1


    b_point = numpy.tile(lb_point,[gridmap[0].astype(int)+1,1]) + numpy.linspace(0., 1., gridmap[0].astype(int)+1).reshape([gridmap[0].astype(int)+1,1])*(lt_point-lb_point)
    gap = numpy.linspace(0., 1., gridmap[1]+1).reshape([gridmap[1].astype(int)+1, 1]) * (rb_point - lb_point)

    for tmp in range(numpy.shape(b_point)[0]):

        if tmp == 0:
            pointmat = b_point[tmp, :] + gap
        else:
            pointmat_tmp =  b_point[tmp, :] + gap
            pointmat = numpy.concatenate((pointmat_tmp,pointmat), axis=0)

    pointmat = pointmat.reshape([gridmap[0].astype(int)+1,gridmap[1].astype(int)+1,2])

    for lon_idx in range(numpy.shape(pointmat)[0]):
        for lat_idx in range(numpy.shape(pointmat)[1]):
            if pointmat[lon_idx][lat_idx][0] > env_border[0] or pointmat[lon_idx][lat_idx][0] < 0 or pointmat[lon_idx][lat_idx][1] < -env_border[1]/2 or pointmat[lon_idx][lat_idx][1] > env_border[1]/2 :
                if lon_idx is 0:
                    if lat_idx is 0:
                        grid_out[lon_idx][lat_idx] = 1
                    elif lat_idx is numpy.shape(pointmat)[1]-1:
                        # pdb.set_trace()
                        grid_out[lon_idx][lat_idx-1] = 1
                    else:
                        grid_out[lon_idx][lat_idx - 1] = 1
                        grid_out[lon_idx][lat_idx] = 1

                elif lon_idx  is numpy.shape(pointmat)[0]-1:
                    if lat_idx is 0:
                        grid_out[lon_idx-1][lat_idx] = 1
                    elif lat_idx is numpy.shape(pointmat)[1]-1:
                        grid_out[lon_idx-1][lat_idx-1] = 1
                    else:
                        grid_out[lon_idx-1][lat_idx - 1] = 1
                        grid_out[lon_idx-1][lat_idx] = 1

                else:
                    if lat_idx is 0:
                        grid_out[lon_idx-1][lat_idx] = 1
                        grid_out[lon_idx][lat_idx] = 1
                    elif lat_idx is numpy.shape(pointmat)[1]-1:
                        grid_out[lon_idx-1][lat_idx-1] = 1
                        grid_out[lon_idx][lat_idx - 1] = 1
                    else:
                        grid_out[lon_idx-1][lat_idx - 1] = 1
                        grid_out[lon_idx-1][lat_idx] = 1
                        grid_out[lon_idx][lat_idx - 1] = 1
                        grid_out[lon_idx][lat_idx] = 1

    return grid_out, pointmat


def step(state,action,del_t,state_range,ego_idx):
    # state = numpy.array([x,y,v,angle])
    # action = numpy.array([accel, del_angle])
    # del_t = time difference

    velo = state[2]
    angle = state[3]

    velo = velo + del_t*action[0]
    angle = angle + del_t*action[1]


    if velo > state_range[0][0]:
        velo = state_range[0][0]
    if velo < state_range[0][1]:
        velo = state_range[0][1]

    if numpy.abs(angle) > state_range[1]:
        angle = numpy.sign(angle)* state_range[1]

    state[0] = state[0] + (del_t * velo * numpy.cos(angle))*(1-ego_idx)
    state[1] = state[1] + del_t * velo * numpy.sin(angle)
    state[2] = velo
    state[3] = angle


    return state



def surveh_model(obstacles,lane_vec,ego_action):
    # state : [numpy.array([x,y,velocity,angle])]
    obs_action = []
    for idx in range(len(obstacles)):
        indices = [i for i,j in enumerate(lane_vec) if lane_vec[idx] == j]
        # pdb.set_trace()
        for obs_idx in indices:
            if obstacles[idx][-1] == 0:
                if (obstacles[idx][0]  != obstacles[obs_idx][0])  and (0. > obstacles[idx][0] - obstacles[obs_idx][0] > -10.) :
                    obstacles[idx][0] = obstacles[obs_idx][0]-10.
                    obs_action.append(numpy.array([-ego_action,0.0]))

                else:
                    obs_action.append(numpy.array([0.05*random.random()-ego_action, 0.0]))

            else:
                if (obstacles[idx][0]  != obstacles[obs_idx][0])  and (0. < obstacles[idx][0] - obstacles[obs_idx][0] < 10.) :
                    obstacles[idx][0] = obstacles[obs_idx][0] + 10.
                    obs_action.append(numpy.array([ego_action,0.0]))

                else:
                    obs_action.append(numpy.array([0.00*random.random()+ego_action, 0.0]))

    return obs_action, obstacles

def chk_done(ego_state, obstacles,safety_radius,env_boundary):
    ego_loc = ego_state[0:2]

    if ego_loc[0] > env_boundary[0] or numpy.abs(ego_loc[1]) > env_boundary[1]/2 :
        done = 1
        reward =0
    else:
        done = 0
        reward = 0

        for idx in range(len(obstacles)) :
            if numpy.sqrt(numpy.linalg.norm(ego_loc-obstacles[idx][0:2])) < safety_radius:
                done = 1
                reward = -ego_state[2]/10.

    return done, reward


class CarSprite(pygame.sprite.Sprite):
    def __init__(self, image, position,env_res):
        pygame.sprite.Sprite.__init__(self)
        self.src_image = pygame.transform.scale(pygame.transform.flip(pygame.image.load(image), 0, 1), (int(numpy.ceil(10*env_res)*0.55),int(numpy.ceil(10*env_res))))
        self.position = position


    def update(self,position,angle):
        self.image = self.src_image
        self.image = pygame.transform.rotate(self.image, -angle/numpy.pi*180)
        self.rect = self.image.get_rect()
        self.rect.center = position


def init_obs(num_obs,env_size,num_lane,lane_dir):
    lane_idx = random.randrange(0, num_lane)
    init_num = random.randrange(1,num_obs+1)
    obstacles = []
    lane_vec = []
    for idx in range(init_num):

        lane_idx = random.randrange(0, num_lane)
        dir = lane_dir[lane_idx]
        lane_vec.append(lane_idx)

        obs = numpy.array(
            [env_size[0]+random.random() - (1. + dir) / 2 * env_size[0], ((1. + 2 * lane_idx) / (2 * num_lane) - 0.5) * env_size[1],
            10.+ random.random() * 10., (1. - dir) / 2 * numpy.pi])

        obstacles.append(obs)

    return obstacles,lane_vec




