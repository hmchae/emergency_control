from collections import namedtuple
from itertools import count
import math
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


Transition = namedtuple(Transition, ('state', 'action','next_state','reward' ))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DDPG(nn.Module):
    def __init__(self, nnStruct):
        super(DDPG,self).__init__()
        nnStruct[0] = network_type
        nnStruct[1] = network_spec

        net_module = nn.ModuleList()
        for idx, layer_type in enumerate(network_type):
            if layer_type is 'linear':
                net_module.append(nn.Linear(network_spec[0],network_spec[1]))

            elif layer_type is 'conv':
                net_module.append(nn.Conv2d(network_spec[0],network_spec[1],network_spec[2]))


                # self.model = torch.nn.Sequential()

