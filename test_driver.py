#Driver code

from environment import Environment
import numpy as np

env = Environment()
print(env.place_horizontal_barrier([0,0]))
print(env.horizontal_barricades)
print(env.return_valid_moves(0))
print(env.check_win())
