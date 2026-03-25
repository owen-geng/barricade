#PLAN:
#Representation - game is an MDP since the board state can be represented entirely irrespective of past:
#7 layers on input: horizontal barricades, vertical barricades, colour planes representing barricade count, 2 player positions and who's turn it is (0 or 1).
#Action space: 12 directional moves + (n-1)^2 h-barricades + (n-1)^2 v-barricades
#  7x7: 12 + 36 + 36 = 84    9x9: 12 + 64 + 64 = 140

import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import time
from environment import Environment, _jit_path_length


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
    

class ReplayBuffer():
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, next_mask, done, n_actual):
        self.buffer.append((state, action, reward, next_state, next_mask, done, n_actual))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, next_mask, done, n_actual = zip(*batch)
        return (torch.stack(state),
                torch.tensor(action),
                torch.tensor(reward, dtype=torch.float32),
                torch.stack(next_state),
                torch.stack(next_mask),
                torch.tensor(done, dtype=torch.float32),
                torch.tensor(n_actual, dtype=torch.float32))
    

class Agent:
    def __init__(self, n, barrier_count):
        self._n = n
        self._barrier_count = barrier_count
        output_dim = 12 + 2*(n-1)*(n-1)
        self.policy_net = torch.compile(AgentNet(n, output_dim).to(device))
        self.target_net = torch.compile(AgentNet(n, output_dim).to(device))
        self.target_net.load_state_dict(self.policy_net.state_dict()) #Target is equiv to policy at init

        self.optimiser = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.buffer = ReplayBuffer()
        self.eps = 1.0 #Epsilon greedy
        self.eps_min = 0.05
        self.eps_decay =  0.999984
        self.gamma = 0.99
        self.batch_size = 1024
        self.train_freq = 4
        self.target_update_freq = 500 #Updates target every 500 steps.
        self.n_steps = 10
        self.n_step_buf = deque()

    def push_transition(self, state, action, reward, next_state, next_mask, done):
        self.n_step_buf.append((state, action, reward, next_state, next_mask, done))
        if len(self.n_step_buf) == self.n_steps:
            self._commit_oldest()
        if done:
            while self.n_step_buf:  # flush partial window at episode end
                self._commit_oldest()

    def _commit_oldest(self):
        """Compute n-step return from the oldest transition in the window and push to replay buffer."""
        G = 0.0
        final_ns = None
        final_mask = None
        final_done = False
        k = 0
        for i, (_, _, r, ns, nm, d) in enumerate(self.n_step_buf):
            G += (self.gamma ** i) * r
            final_ns = ns
            final_mask = nm
            k = i + 1
            if d:
                final_done = True
                break
        s0, a0 = self.n_step_buf[0][0], self.n_step_buf[0][1]
        self.buffer.push(s0, a0, G, final_ns, final_mask, final_done, k)
        self.n_step_buf.popleft()

    def select_action(self, state, mask):
        if random.random() < self.eps:
            #Take random action
            valid = mask.nonzero().flatten()
            return valid[random.randrange(len(valid))].item()
        with torch.no_grad():
            q = self.policy_net(state.unsqueeze(0).to(device))[0]
            q[~mask.to(device)] = -float('inf') #mask invalid actions
            return q.argmax().item()

    def train_step(self, step):
        if len(self.buffer.buffer) < self.batch_size: #Buffer can't fill a full batch yet
            return
        state, action, reward, next_state, next_mask, done, n_actual = self.buffer.sample(self.batch_size)
        state, next_state = state.to(device), next_state.to(device)
        action, reward, done = action.to(device), reward.to(device), done.to(device)
        next_mask = next_mask.to(device)
        n_actual = n_actual.to(device)

        q_vals = self.policy_net(state).gather(1, action.unsqueeze(1)).squeeze()

        with torch.no_grad():
            # Double DQN: policy_net selects action, target_net evaluates it.
            # Mask invalid actions before argmax to avoid bootstrapping from illegal moves.
            next_q_policy = self.policy_net(next_state)
            next_q_policy[~next_mask] = -float('inf')
            next_actions = next_q_policy.argmax(1, keepdim=True)
            next_q = self.target_net(next_state).gather(1, next_actions).squeeze(1)
            targets = reward + (self.gamma ** n_actual) * next_q * (1-done) #n-step Bellman update
        
        loss = nn.functional.smooth_l1_loss(q_vals, targets)
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10) #Gradient clipping.
        self.optimiser.step()

        self.eps = max(self.eps_min, self.eps*self.eps_decay)
        if step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            torch.save({'state_dict': {k.replace('_orig_mod.', ''): v for k, v in self.policy_net.state_dict().items()},
                        'n': self._n, 'barrier_count': self._barrier_count}, 'dqn.pt')



#Training loop

def _eval_vs_random(agent, n, barrier_count, games=100):
    """Play `games` matches: agent as p0 for half, p1 for half. Returns win rate."""
    env = Environment(n, barrier_count)
    score = 0.0
    for game in range(games):
        agent_side = game % 2  # alternate sides
        env.__init__(n, barrier_count)
        done = False
        turn = 0
        while not done and turn < MAX_TURNS:
            p = env.player_turn
            mask = env.return_action_mask()
            if p == agent_side:
                with torch.no_grad():
                    q = agent.policy_net(env.return_state_representation().unsqueeze(0).to(device))[0]
                    q[~mask.to(device)] = -float('inf')
                    action = q.argmax().item()
            else:
                valid = mask.nonzero().flatten()
                action = valid[random.randrange(len(valid))].item()
            env.agent_action_function(action)
            turn += 1
            winner = env.check_win()
            if winner is not None:
                if winner == agent_side:
                    score += 1.0
                done = True
        if not done:  # timeout = draw
            score += 0.5
    return score / games



def _path_length(env, player):
    loc = env.p1loc if player == 0 else env.p2loc
    target = env.n - 1 if player == 0 else 0
    return _jit_path_length(env.horizontal_barricades, env.vertical_barricades,
                            env.n, loc[0], loc[1], target)

MAX_TURNS = 200
POOL_SIZE = 15        # max past checkpoints to keep
POOL_ADD_FREQ = 500   # add learner to pool every N episodes
OPP_SWAP_FREQ = 50    # sample a new opponent from pool every N episodes


def _pool_select_action(net, state, mask):
    """Greedy action from a frozen opponent network."""
    with torch.no_grad():
        q = net(state.unsqueeze(0).to(device))[0]
        q[~mask.to(device)] = -float('inf')
        return q.argmax().item()


def train(episodes=50000, n=7, barrier_count=10):
    env = Environment(n,barrier_count)
    learner = Agent(n, barrier_count)  # only agent that trains
    step = 0

    # Opponent pool: list of state_dicts. Initialised with a random-weight snapshot.
    # torch.compile adds "_orig_mod." prefix — strip it so uncompiled opp_net can load.
    def _snapshot(net):
        return {k.replace('_orig_mod.', ''): v.clone() for k, v in net.state_dict().items()}

    opp_net = AgentNet(n, 12 + 2*(n-1)*(n-1)).to(device)
    pool = [_snapshot(learner.policy_net)]
    opp_net.load_state_dict(pool[-1])
    opp_net.eval()

    t0 = time.time()
    recent_turns = deque(maxlen=100)
    recent_timeouts = deque(maxlen=100)
    for ep in range(episodes):
        ep_start = time.time()

        # Periodically add learner snapshot to pool
        if ep > 0 and ep % POOL_ADD_FREQ == 0:
            pool.append(_snapshot(learner.policy_net))
            if len(pool) > POOL_SIZE:
                pool.pop(0)

        # Swap opponent every OPP_SWAP_FREQ episodes — sample uniformly from pool
        if ep % OPP_SWAP_FREQ == 0:
            opp_net.load_state_dict(random.choice(pool))
            opp_net.eval()

        learner_side = ep % 2  # alternate which side the learner plays
        env.__init__(n, barrier_count)
        done = False
        dist = [_path_length(env, 0), _path_length(env, 1)]
        turn = 0
        cycle_penalty = 0.03 * max(0.0, 1.0 - ep / 20000)
        history = deque(maxlen=6)

        # If learner is player 1, opponent (player 0) must move first
        if learner_side == 1:
            opp_mask = env.return_action_mask()
            if ep < 1000:
                valid = opp_mask.nonzero().flatten()
                opp_action = valid[random.randrange(len(valid))].item()
            else:
                opp_state = env.return_state_representation()
                opp_action = _pool_select_action(opp_net, opp_state, opp_mask)
            env.agent_action_function(opp_action)
            turn += 1
            if env.check_win() is not None:
                done = True
            dist = [_path_length(env, 0), _path_length(env, 1)]

        state = env.return_state_representation()

        while not done and turn < MAX_TURNS:
            # ---- Learner's half-turn ----
            assert env.player_turn == learner_side
            learner_state = state
            learner_mask  = env.return_action_mask()
            action = learner.select_action(learner_state, learner_mask)

            env.agent_action_function(action)
            turn += 1
            winner = env.check_win()
            done = winner is not None

            if done:
                reward = 1.0 if winner == learner_side else -1.0
                next_state = env.return_state_representation()
                next_mask  = env.return_action_mask()
                learner.push_transition(learner_state, action, reward, next_state, next_mask, True)
                if step % learner.train_freq == 0:
                    learner.train_step(step)
                step += 1
                state = next_state
                break

            new_dist = [_path_length(env, 0), _path_length(env, 1)]
            my_progress  = (dist[learner_side]     - new_dist[learner_side])     * 0.05
            opp_set_back = (new_dist[1-learner_side] - dist[1-learner_side])     * 0.01
            reward = my_progress + opp_set_back - 0.002
            dist = new_dist
            new_pos = tuple(env.p1loc if learner_side == 0 else env.p2loc)
            if new_pos in history:
                reward -= cycle_penalty
            history.append(new_pos)

            # ---- Opponent's half-turn ----
            if turn >= MAX_TURNS:
                # Timeout mid-ply: flush with current state as bootstrap
                next_state = env.return_state_representation()
                next_mask  = env.return_action_mask()
                learner.push_transition(learner_state, action, reward, next_state, next_mask, False)
                if step % learner.train_freq == 0:
                    learner.train_step(step)
                step += 1
                state = next_state
                done = False  # let outer loop exit via turn >= MAX_TURNS check
                break

            opp_mask = env.return_action_mask()
            if ep < 1000:
                valid = opp_mask.nonzero().flatten()
                opp_action = valid[random.randrange(len(valid))].item()
            else:
                opp_state = env.return_state_representation()
                opp_action = _pool_select_action(opp_net, opp_state, opp_mask)

            env.agent_action_function(opp_action)
            turn += 1
            winner = env.check_win()
            done = winner is not None

            if done:
                reward += -1.0  # opponent won — override shaping with terminal signal
                reward  = -1.0

            new_dist = [_path_length(env, 0), _path_length(env, 1)]
            if not done:
                dist = new_dist

            # next_state is now the start of the learner's next turn — max Q is correct
            next_state = env.return_state_representation()
            next_mask  = env.return_action_mask()

            learner.push_transition(learner_state, action, reward, next_state, next_mask, done)
            if step % learner.train_freq == 0:
                learner.train_step(step)
            step += 1
            state = next_state

        # Flush leftover n-step transitions on timeout — prevents cross-episode contamination
        if not done:
            while learner.n_step_buf:
                learner._commit_oldest()

        ep_time = time.time() - ep_start
        recent_turns.append(turn)
        recent_timeouts.append(1 if not done else 0)
        avg_turns = sum(recent_turns) / len(recent_turns)
        timeout_rate = sum(recent_timeouts) / len(recent_timeouts)
        if ep < 200:
            print(f"Ep {ep:>4}  {ep_time*1000:.0f}ms  turns={turn}  avg={avg_turns:.0f}  timeout={timeout_rate:.0%}  eps={learner.eps:.3f}  cycle_pen={cycle_penalty:.4f}  pool={len(pool)}")
        elif ep % 100 == 0:
            elapsed = time.time() - t0
            eps_per_sec = ep / elapsed
            remaining = (episodes - ep) / eps_per_sec
            print(f"Ep {ep:>6}/{episodes}  {eps_per_sec:.1f} ep/s  "
                  f"elapsed {elapsed/60:.1f}m  remaining ~{remaining/60:.1f}m  "
                  f"avg_turns={avg_turns:.0f}  timeout={timeout_rate:.0%}  eps {learner.eps:.3f}  cycle_pen={cycle_penalty:.4f}  pool={len(pool)}")

        if ep > 0 and ep % 1000 == 0:
            wr = _eval_vs_random(learner, n, barrier_count)
            torch.save({'state_dict': {k.replace('_orig_mod.', ''): v for k, v in learner.policy_net.state_dict().items()},
                        'n': n, 'barrier_count': barrier_count}, 'dqn.pt')
            print(f"  [eval ep {ep}] win rate vs random: {wr:.0%}  (saved)")
