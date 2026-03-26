import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import numpy as np
import time
import random
from environment import Environment

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class AlphaZeroNet(nn.Module):
    # Input: 6-channel canonical state (no turn plane)
    # Outputs: policy logits over action_dim, scalar value in [-1, 1]
    def __init__(self, n, action_dim, channels=128, n_blocks=8):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(6, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(n_blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * n * n, action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n * n, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        h = self.body(self.stem(x.float()))
        return self.policy_head(h), self.value_head(h).squeeze(-1)


# ── MCTS ─────────────────────────────────────────────────────────────────────

class MCTSNode:
    def __init__(self, prior: np.ndarray, player: int):
        self.player = player           # whose turn at this node
        self.P = prior                 # (action_dim,) masked+renormalised priors
        self.N = np.zeros_like(prior)  # visit counts per action
        self.W = np.zeros_like(prior)  # total backed-up value per action
        self.children = {}             # canonical action_id -> MCTSNode

    def is_leaf(self):
        return len(self.children) == 0

    def Q(self):
        return np.divide(self.W, self.N, out=np.zeros_like(self.W), where=self.N > 0)


def select_action(node: MCTSNode, c_puct: float = 1.5) -> int:
    sqrt_total = np.sqrt(node.N.sum() + 1)
    puct = node.Q() + c_puct * node.P * sqrt_total / (1 + node.N)
    puct[node.P == 0] = -np.inf  # guard: invalid actions have P=0; block even when all Q<0
    return int(np.argmax(puct))


def _get_prior(env, net, action_dim) -> tuple:
    """Run network on current env state. Returns (masked prior np array, value float)."""
    state = env.return_canonical_state_representation().unsqueeze(0).to(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
        logits, value = net(state)
    logits = logits[0].float()

    # Mask invalid actions: for each canonical action i, check whether its
    # corresponding real action is valid. convert_canonical_action maps i -> real,
    # and is its own inverse, so real_mask[convert_canonical_action(i)] is correct.
    real_mask = torch.from_numpy(env.return_valid_actions_RL().astype(bool))
    canonical_mask = real_mask[[env.convert_canonical_action(i) for i in range(action_dim)]]

    logits[~canonical_mask] = -float('inf')
    prior = torch.softmax(logits, dim=0).cpu().numpy()
    return prior, value.item()


def _backup(path: list, value: float):
    for node, action in reversed(path):
        node.W[action] += value
        node.N[action] += 1
        value = -value


VIRTUAL_LOSS = 1.0

def _backup_vl(path: list, value: float):
    """Backup for batched MCTS. During selection each edge had W[action] -= VL
    and N[action] += 1 already, so we only add (value + VL) to W."""
    for node, action in reversed(path):
        node.W[action] += value + VIRTUAL_LOSS
        value = -value


def run_mcts(root_env: Environment, root_node: MCTSNode, net: AlphaZeroNet,
             n_simulations: int = 400, c_puct: float = 1.5, verbose: bool = False):
    # Root always has priors set — selection always starts from root.
    # Children are created lazily on first visit to an edge.
    action_dim = len(root_node.P)

    for sim in range(n_simulations):
        if verbose:
            print(f'\r    sim {sim+1}/{n_simulations}', end='', flush=True)

        env = root_env.clone()
        node = root_node
        path = []

        while True:
            action = select_action(node, c_puct)
            path.append((node, action))
            env.agent_action_function(env.convert_canonical_action(action))

            winner = env.check_win()
            if winner is not None:
                # Player who just acted won (+1), or draw (0.0)
                val = 0.0 if winner == 2 else (1.0 if winner == path[-1][0].player else -1.0)
                _backup(path, val)
                break

            if action not in node.children:
                # First visit to this edge — expand the child
                prior, value = _get_prior(env, net, action_dim)
                node.children[action] = MCTSNode(prior, env.player_turn)
                # value is from the child's player perspective; negate for the parent
                _backup(path, -value)
                break

            node = node.children[action]


def run_mcts_batched(root_env: Environment, root_node: MCTSNode, net: AlphaZeroNet,
                     n_simulations: int = 400, leaf_batch_size: int = 8,
                     c_puct: float = 1.5):
    """Batched-leaf MCTS: collect leaf_batch_size simulation paths, evaluate all
    their leaves in one GPU forward pass, then backup all paths.  Virtual loss
    is applied during selection so parallel simulations diverge."""
    action_dim = len(root_node.P)
    sim = 0

    while sim < n_simulations:
        bs = min(leaf_batch_size, n_simulations - sim)
        sim += bs

        paths          = []   # one path per simulation in this batch
        leaf_states    = []   # canonical states for non-terminal leaves
        leaf_envs      = []   # envs at non-terminal leaves (for masking)
        terminal_vals  = []   # None = needs NN eval; float = terminal value

        # ── Phase 1: selection with virtual loss ──────────────────────────────
        for _ in range(bs):
            env  = root_env.clone()
            node = root_node
            path = []
            term = None

            while True:
                action = select_action(node, c_puct)
                # Apply virtual loss so subsequent sims in this batch diverge
                node.W[action] -= VIRTUAL_LOSS
                node.N[action] += 1
                path.append((node, action))
                env.agent_action_function(env.convert_canonical_action(action))

                winner = env.check_win()
                if winner is not None:
                    term = 0.0 if winner == 2 else (1.0 if winner == path[-1][0].player else -1.0)
                    break

                if action not in node.children:
                    # Unexplored leaf — queue for NN evaluation
                    leaf_states.append(env.return_canonical_state_representation())
                    leaf_envs.append(env)
                    break

                node = node.children[action]

            paths.append(path)
            terminal_vals.append(term)

        # ── Phase 2: batch NN evaluation ──────────────────────────────────────
        if leaf_states:
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda'):
                logits_b, values_b = net(torch.stack(leaf_states).to(device))
            logits_b = logits_b.float().cpu()
            values_b = values_b.float().cpu()

        # ── Phase 3: expand + backup ──────────────────────────────────────────
        leaf_idx = 0
        for i, path in enumerate(paths):
            if terminal_vals[i] is not None:
                _backup_vl(path, terminal_vals[i])
            else:
                env    = leaf_envs[leaf_idx]
                logits = logits_b[leaf_idx].clone()

                real_mask  = torch.from_numpy(env.return_valid_actions_RL().astype(bool))
                canon_mask = real_mask[[env.convert_canonical_action(j)
                                        for j in range(action_dim)]]
                logits[~canon_mask] = -float('inf')
                prior = torch.softmax(logits, dim=0).numpy()

                # Attach child (a concurrent sim in this batch may have already
                # expanded the same edge; overwriting is benign but slightly wasteful)
                last_node, last_action = path[-1]
                last_node.children[last_action] = MCTSNode(prior, env.player_turn)

                _backup_vl(path, -values_b[leaf_idx].item())
                leaf_idx += 1


# ── Self-play ─────────────────────────────────────────────────────────────────

def _add_dirichlet_noise(root_node: MCTSNode, epsilon: float = 0.25, alpha: float = 0.07):
    """Mix Dirichlet noise into the root prior over valid actions only.
    epsilon: weight on noise (0.25 is standard AlphaZero).
    alpha: Dirichlet concentration — smaller = spikier noise. Use ~10/action_dim."""
    valid = root_node.P > 0
    n_valid = valid.sum()
    if n_valid == 0:
        return
    noise = np.random.dirichlet(np.full(n_valid, alpha))
    root_node.P[valid] = (1 - epsilon) * root_node.P[valid] + epsilon * noise

def self_play_game(net: AlphaZeroNet, n: int, barrier_count: int,
                   n_simulations: int = 200, temp_cutoff: int = 25,
                   leaf_batch_size: int = 1, verbose: bool = False) -> list:
    """Play one full game. Returns list of (canonical_state, pi, player) tuples.
    z is assigned after the game ends.
    leaf_batch_size=1 uses sequential run_mcts; >1 uses run_mcts_batched."""
    action_dim = 12 + 2 * (n - 1) ** 2
    env = Environment(n, barrier_count)

    # Build root node
    prior, _ = _get_prior(env, net, action_dim)
    root_node = MCTSNode(prior, env.player_turn)

    trajectory = []  # (state_tensor, pi, player)
    move = 0

    while env.check_win() is None:
        if verbose:
            print(f'\n  move {move+1}', flush=True)
        _add_dirichlet_noise(root_node)
        if leaf_batch_size > 1:
            run_mcts_batched(env, root_node, net, n_simulations, leaf_batch_size)
        else:
            run_mcts(env, root_node, net, n_simulations, verbose=verbose)
        if verbose:
            print()

        counts = root_node.N.copy()
        if move < temp_cutoff:
            counts = counts ** 1.0  # temp=1: sample proportional to visits
            pi = counts / counts.sum()
            action = np.random.choice(len(pi), p=pi)
        else:
            action = int(np.argmax(counts))  # temp→0: greedy
            pi = np.zeros_like(counts)
            pi[action] = 1.0

        trajectory.append((
            env.return_canonical_state_representation(),
            pi,
            env.player_turn,
        ))

        env.agent_action_function(env.convert_canonical_action(action))
        move += 1

        # Tree reuse — re-root at chosen child
        root_node = root_node.children.get(action)
        if root_node is None:
            prior, _ = _get_prior(env, net, action_dim)
            root_node = MCTSNode(prior, env.player_turn)

    winner = env.check_win()
    return [
        (state, pi, 0.0 if winner == 2 else (1.0 if player == winner else -1.0))
        for state, pi, player in trajectory
    ]


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 500_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, pi, z):
        self.buffer.append((state.cpu(), pi, z))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, pis, zs = zip(*batch)
        return (
            torch.stack(states).to(device),
            torch.tensor(np.array(pis), dtype=torch.float32).to(device),
            torch.tensor(zs, dtype=torch.float32).to(device),
        )

    def __len__(self):
        return len(self.buffer)


# ── Training ──────────────────────────────────────────────────────────────────

def train_step(net: AlphaZeroNet, optimizer, batch):
    states, pi_targets, z_targets = batch
    policy_logits, value = net(states)

    log_pi = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(pi_targets * log_pi).sum(dim=1).mean()
    value_loss = F.mse_loss(value, z_targets)
    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()
    return loss.item(), policy_loss.item(), value_loss.item()


def _eval_vs_random(net: AlphaZeroNet, n: int, barrier_count: int,
                    n_simulations: int = 100, games: int = 40,
                    leaf_batch_size: int = 1) -> float:
    action_dim = 12 + 2 * (n - 1) ** 2
    score = 0.0
    for game in range(games):
        agent_side = game % 2
        env = Environment(n, barrier_count)

        prior, _ = _get_prior(env, net, action_dim)
        root = MCTSNode(prior, env.player_turn)

        while env.check_win() is None:
            if env.player_turn == agent_side:
                if leaf_batch_size > 1:
                    run_mcts_batched(env, root, net, n_simulations, leaf_batch_size)
                else:
                    run_mcts(env, root, net, n_simulations)
                canonical_action = int(np.argmax(root.N))
                env.agent_action_function(env.convert_canonical_action(canonical_action))
                # Tree reuse — re-root at chosen child
                root = root.children.get(canonical_action)
                if root is None:
                    prior, _ = _get_prior(env, net, action_dim)
                    root = MCTSNode(prior, env.player_turn)
            else:
                real_mask = env.return_valid_actions_RL().astype(bool)
                valid = np.where(real_mask)[0]
                real_action = int(np.random.choice(valid))
                env.agent_action_function(real_action)
                # Random move isn't in the tree — rebuild root for next MCTS turn
                prior, _ = _get_prior(env, net, action_dim)
                root = MCTSNode(prior, env.player_turn)

        winner = env.check_win()
        if winner == agent_side:
            score += 1.0
        elif winner == 2:  # draw
            score += 0.5
    return score / games


def train(n: int = 7, barrier_count: int = 10, n_iterations: int = 200,
          games_per_iter: int = 20, grad_steps_per_iter: int = 100,
          batch_size: int = 512, n_simulations: int = 400,
          leaf_batch_size: int = 1, checkpoint: str = None,
          buffer_capacity: int = None):

    import os
    if checkpoint is None:
        checkpoint = f'alphazero_{n}_by_{n}.pt'
    action_dim = 12 + 2 * (n - 1) ** 2
    raw_net = AlphaZeroNet(n, action_dim).to(device)   # kept for state_dict saving
    optimizer = optim.AdamW(raw_net.parameters(), lr=1e-3, weight_decay=1e-4)
    if buffer_capacity is None:
        # ~500 games worth of states; avg game length scales with board area
        avg_game_len = 2 * barrier_count + n * n // 2
        buffer_capacity = avg_game_len * 500
    buffer = ReplayBuffer(capacity=buffer_capacity)
    start_iter = 0

    if os.path.exists(checkpoint):
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        raw_net.load_state_dict(ck['state_dict'])
        if 'optimizer_state_dict' in ck:
            optimizer.load_state_dict(ck['optimizer_state_dict'])
        start_iter = ck.get('iteration', 0)
        if 'buffer' in ck:
            for item in ck['buffer']:
                buffer.push(*item)
            print(f"Resumed from {checkpoint}  (iteration {start_iter}, buffer={len(buffer)})")
        else:
            print(f"Resumed from {checkpoint}  (iteration {start_iter})")
    else:
        print(f"No checkpoint found at {checkpoint}, starting from scratch.")

    net = torch.compile(raw_net)   # compiled net used for all inference/training

    t0 = time.time()
    iteration_count = start_iter
    for it in range(n_iterations):
        print(f"Completed iteration: {iteration_count}")
        iteration_count += 1
        net.eval()
        game_count = 0
        for _ in range(games_per_iter):
            game_count += 1
            print(f"Iteration: {iteration_count}. Game count: {game_count}")
            game_data = self_play_game(net, n, barrier_count, n_simulations,
                                       leaf_batch_size=leaf_batch_size)
            for state, pi, z in game_data:
                buffer.push(state, pi, z)

        total_loss = 0.0
        print(f"Length of buffer: {len(buffer)}")
        if len(buffer) >= batch_size:
            net.train()
            for _ in range(grad_steps_per_iter):
                batch = buffer.sample(batch_size)
                loss, _, _ = train_step(net, optimizer, batch)
                total_loss += loss

        elapsed = time.time() - t0
        print(f"iter {iteration_count}/{start_iter + n_iterations}  buffer={len(buffer)}  "
              f"loss={total_loss/grad_steps_per_iter:.4f}  elapsed={elapsed/60:.1f}m")

        torch.save({
            'state_dict': raw_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'iteration': iteration_count,
            'n': n,
            'barrier_count': barrier_count,
            'buffer': list(buffer.buffer),
        }, checkpoint)
        net.eval()
