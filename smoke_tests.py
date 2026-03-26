"""Smoke tests for the MCTS pipeline (n=5, barrier_count=5)."""
import numpy as np
from environment import Environment
from MCTS_agent import (
    AlphaZeroNet, MCTSNode, _get_prior, _backup, run_mcts, self_play_game, device
)

N = 5
BARRIER_COUNT = 5
ACTION_DIM = 12 + 2 * (N - 1) ** 2  # 44


def make_net():
    net = AlphaZeroNet(N, ACTION_DIM).to(device)
    net.eval()
    return net


# ── 1. _get_prior ─────────────────────────────────────────────────────────────

def test_get_prior():
    print("--- test_get_prior ---")
    net = make_net()
    env = Environment(N, BARRIER_COUNT)
    prior, value = _get_prior(env, net, ACTION_DIM)

    assert prior.shape == (ACTION_DIM,), f"prior shape {prior.shape}"
    assert abs(prior.sum() - 1.0) < 1e-5, f"prior sums to {prior.sum()}"
    assert -1.0 <= value <= 1.0, f"value {value} out of range"

    # Invalid actions must have zero prior
    real_mask = env.return_valid_actions_RL().astype(bool)
    canonical_mask = np.array([
        real_mask[env.convert_canonical_action(i)] for i in range(ACTION_DIM)
    ])
    assert (prior[~canonical_mask] == 0).all(), "invalid actions have non-zero prior"

    print(f"  prior sum={prior.sum():.6f}  value={value:.4f}  "
          f"valid={canonical_mask.sum()}  invalid zeroed={( ~canonical_mask).sum()}")
    print("PASS  _get_prior")


# ── 2. _backup ────────────────────────────────────────────────────────────────

def test_backup():
    print("\n--- test_backup ---")
    prior = np.ones(ACTION_DIM) / ACTION_DIM
    node0 = MCTSNode(prior.copy(), player=0)
    node1 = MCTSNode(prior.copy(), player=1)
    node2 = MCTSNode(prior.copy(), player=0)

    path = [(node0, 3), (node1, 7), (node2, 1)]
    _backup(path, 1.0)  # node2 (player 0) just won

    # node2: value=+1 from its perspective
    assert node2.N[1] == 1 and abs(node2.W[1] - 1.0) < 1e-9
    # node1: value flipped once -> -1
    assert node1.N[7] == 1 and abs(node1.W[7] - (-1.0)) < 1e-9
    # node0: value flipped twice -> +1
    assert node0.N[3] == 1 and abs(node0.W[3] - 1.0) < 1e-9

    print(f"  node0.W[3]={node0.W[3]:.1f}  node1.W[7]={node1.W[7]:.1f}  node2.W[1]={node2.W[1]:.1f}")
    print("PASS  _backup")


# ── 3. run_mcts ───────────────────────────────────────────────────────────────

def test_run_mcts():
    print("\n--- test_run_mcts ---")
    net = make_net()
    env = Environment(N, BARRIER_COUNT)
    prior, _ = _get_prior(env, net, ACTION_DIM)
    root = MCTSNode(prior, env.player_turn)

    n_sims = 50
    run_mcts(env, root, net, n_simulations=n_sims, verbose=True)
    print()

    total_visits = int(root.N.sum())
    assert total_visits > 0, "no visits recorded"
    assert total_visits <= n_sims, f"visits {total_visits} exceed n_sims {n_sims}"

    # select_action should never return an invalid action
    real_mask = env.return_valid_actions_RL().astype(bool)
    canonical_mask = np.array([
        real_mask[env.convert_canonical_action(i)] for i in range(ACTION_DIM)
    ])
    best = int(np.argmax(root.N))
    assert canonical_mask[best], f"best action {best} is invalid"

    print(f"  total_visits={total_visits}  best_action={best}  "
          f"best_N={root.N[best]:.0f}  children={len(root.children)}")
    print("PASS  run_mcts")


# ── 4. self_play_game ─────────────────────────────────────────────────────────

def test_self_play_game():
    print("\n--- test_self_play_game ---")
    net = make_net()
    data = self_play_game(net, N, BARRIER_COUNT, n_simulations=50, verbose=True)
    print()

    assert len(data) > 0, "empty trajectory"
    states, pis, zs = zip(*data)

    for i, (s, pi, z) in enumerate(zip(states, pis, zs)):
        assert s.shape == (6, N, N), f"step {i}: state shape {s.shape}"
        assert abs(pi.sum() - 1.0) < 1e-5, f"step {i}: pi sums to {pi.sum()}"
        assert z in (1.0, -1.0, 0.0), f"step {i}: z={z}"

    print(f"  game_length={len(data)}  z_counts={{+1: {sum(z==1 for _,_,z in data)}, -1: {sum(z==-1 for _,_,z in data)}, 0: {sum(z==0 for _,_,z in data)}}}")
    print("PASS  self_play_game")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_get_prior()
    test_backup()
    test_run_mcts()
    test_self_play_game()
    print("\nAll smoke tests passed.")
