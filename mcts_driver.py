from MCTS_agent import train

train(
    n               = 9,
    barrier_count   = 10,
    n_iterations    = 5000,
    games_per_iter  = 50,
    grad_steps_per_iter = 400,
    batch_size      = 256,
    n_simulations   = 400,
    leaf_batch_size = 128,
)

"""
train(
    n               = 5,
    barrier_count   = 5,
    n_iterations    = 5000,
    games_per_iter  = 30,
    grad_steps_per_iter = 300,
    batch_size      = 256,
    n_simulations   = 200,
    leaf_batch_size = 64,
)


"""