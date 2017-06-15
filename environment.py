import pygame
import numpy
import math
import pdb
import matplotlib.pyplot as plt


def vehicle_input(ego_state, sensor_range, obstacles, cell_size, env_border):

    # ego_state/obstacles : [x,y,velocity,angle]
    # sensor_range : [longitudinal range, lateral range]

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
            # pdb.set_trace()
            if numpy.linalg.norm(obs_state[0:2]-lb_point) > numpy.linalg.norm(obs_state[0:2]-rb_point):
                dir = 1
            else:
                dir = -1

            loc_y = loc_y*dir
            loc_tmp = numpy.floor(numpy.array([loc_x,loc_y])/numpy.array(cell_size)*numpy.array([-1.,1.])) + numpy.array([gridmap[0],numpy.ceil(gridmap[1]/2)])
            grid_loc = loc_tmp.astype(int).tolist()
            # pdb.set_trace()
            grid_out[grid_loc[0]][grid_loc[1]] = 1

    # pdb.set_trace()
    b_point = numpy.tile(lb_point,[gridmap[0].astype(int)+1,1]) + numpy.linspace(0., 1., gridmap[0].astype(int)+1).reshape([gridmap[0].astype(int)+1,1])*(lt_point-lb_point)
    gap = numpy.linspace(0., 1., gridmap[1]+1).reshape([gridmap[1].astype(int)+1, 1]) * (rb_point - lb_point)
    # point_mat = numpy.array([])

    for tmp in range(numpy.shape(b_point)[0]):

        if tmp == 0:
            pointmat = b_point[tmp, :] + gap
            # pointmat = pointmat[::-1]
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
