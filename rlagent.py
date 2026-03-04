#PLAN:
#Representation - game is an MDP since the board state can be represented entirely irrespective of past:
#7 layers on input: horizontal barricades, vertical barricades, colour planes representing barricade count, 2 player positions and who's turn it is (0 or 1).

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


import matplotlib
import random

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

#seed = 67
#random.seed(seed)
#torch.manual_seed(seed)


class AgentNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        channel, height, width = input_dim
        
        #For the case of barricade, expecting c = 7
        #"""
        if channel != 7:
            raise ValueError(f"Expecting 7 channels, got {channel}.")
        
        #"""
    

    def cnn(self, channel, output_dim):

        l1out = 32
        l1kernel = 8
        l1stride = 4

        l2out = 64
        l2kernel = 4
        l2stride = 2

        l3out = 64
        l3kernel = 3
        l3stride = 1

        return nn.Sequential(
            nn.Conv2d(in_channels=channel, out_channels=l1out,kernel_size=l1kernel,stride=l1stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=l1out, out_channels=l2out,kernel_size=l2kernel,stride=l2stride),
            nn.ReLU(),
            nn.Conv2d(in_channels=l2out, out_channels=l3out,kernel_size=l3kernel,stride=l3stride),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512), #3136 needs to be changed if input dimensions change
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )


class Agent:
    def __init__():
        pass

