#PLAN:
#Representation - game is an MDP since the board state can be represented entirely irrespective of past:
#7 layers on input: horizontal barricades, vertical barricades, colour planes representing barricade count, 2 player positions and who's turn it is (0 or 1).
#Action space: 12 directional moves + (n-1)^2 h-barricades + (n-1)^2 v-barricades
#  7x7: 12 + 36 + 36 = 84    9x9: 12 + 64 + 64 = 140

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

    def __init__(self, n, output_dim):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        size_pass = torch.zeros(1, 7, n, n)
        flat_layer_size = self.cnn(size_pass).shape[1]

        self.head = nn.Sequential(
            nn.Linear(flat_layer_size, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.head(self.cnn(x.float()))

class Agent:
    def __init__(self, n):
        
        pass

