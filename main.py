from environment import Environment
import numpy as np


env = Environment()

env.move_debug(0, [0,0])
env.horizontal_barricades[1][3] = 1 
env.vertical_barricades[2][2] = 1
env.move_debug(0, [1,0])
print(env.return_valid_moves(0))
print(env.return_valid_moves(1))