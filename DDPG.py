from collections import namedtuple
from itertools import count
import math
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np


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

    def __init__(self, h, w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 30, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(30)
        self.conv2 = nn.Conv2d(30, 20, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv3 = nn.Conv2d(20, 10, kernel_size=2)
        self.bn3 = nn.BatchNorm2d(10)

        def conv2d_size_out(size, kernel_size):
            return size - (kernel_size - 1)

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,3),3),2)
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,3),3),2)
        linear_input_size = convw * convh * 10
        self.head = nn.Linear(linear_input_size, 2) # 448 or 512

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return self.head(x.view(x.size(0), -1))
