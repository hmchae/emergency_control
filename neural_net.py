import numpy
import torch
import os
import sys
import torch.nn as nn


class net_module(nn.Module):
    def __init__(network_type, network_spec):
        super(net_module, self).__init__()


        net_string = ''
        for idx, layer_type in enumerate(network_type):
            if layer_type is 'linear':
                net_string += 'nn.Linear(network_spec[0],network_spec[1]),'
                net_string += 'nn.'+ network_specp[-1] + ','
            elif layer_type = is 'conv':
                net


        self.model = torch.nn.Sequential()
