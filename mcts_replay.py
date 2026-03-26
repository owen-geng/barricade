"""MCTS self-play game displayed live in a GUI.
Each move is searched in a background thread so the board updates in real time.

Modes (set VS_RANDOM at the bottom):
  VS_RANDOM = False  —  MCTS (Blue) vs MCTS (Red)  [self-play]
  VS_RANDOM = True   —  MCTS (Blue) vs Random (Red)
"""
import os
import threading
import tkinter as tk
import numpy as np
import torch
from environment import Environment
from MCTS_agent import AlphaZeroNet, MCTSNode, _get_prior, run_mcts, run_mcts_batched, device

def load_net(n=None, path=None):
    if n is None:
        n = N
    if path is None:
        path = f'alphazero_{n}_by_{n}.pt'
    if os.path.exists(path):
        ck = torch.load(path, map_location=device, weights_only=False)
        n, barrier_count = ck['n'], ck['barrier_count']
        action_dim = 12 + 2 * (n - 1) ** 2
        net = AlphaZeroNet(n, action_dim).to(device)
        net.load_state_dict(ck['state_dict'])
        print(f"Loaded {path}  (n={n}, barriers={barrier_count})")
    else:
        print(f"No checkpoint found at {path}, using random weights.")
        barrier_count = BARRIER_COUNT
        action_dim = 12 + 2 * (n - 1) ** 2
        net = AlphaZeroNet(n, action_dim).to(device)
    net.eval()
    return net, n, barrier_count

N = 9
BARRIER_COUNT = 5
N_SIMULATIONS = 400
CELL_SIZE = 80
PAUSE_AFTER_MOVE_MS = 400   # pause between moves once thinking is done


class LiveReplayGUI:
    def __init__(self, root_win, net, n, barrier_count, n_simulations,
                 leaf_batch_size=1, vs_random=False, mcts_player=0):
        self.root_win = root_win
        self.net = net
        self.n = n
        self.n_simulations = n_simulations
        self.leaf_batch_size = leaf_batch_size
        self.action_dim = 12 + 2 * (n - 1) ** 2
        self.vs_random = vs_random
        self.mcts_player = mcts_player          # 0 = MCTS is Blue, 1 = MCTS is Red
        self.random_player = 1 - mcts_player    # the other player plays randomly

        self.env = Environment(n, barrier_count)
        prior, _ = _get_prior(self.env, net, self.action_dim)
        self.root_node = MCTSNode(prior, self.env.player_turn)
        self.move_num = 0
        self.values = [0.0, 0.0]  # expected reward per player, from their own perspective

        cs = CELL_SIZE
        self.canvas = tk.Canvas(root_win, width=n * cs, height=n * cs, bg="white")
        self.canvas.pack()

        frame1 = tk.Frame(root_win)
        frame1.pack()
        self.status = tk.Label(frame1, text="", font=("Arial", 12))
        self.status.pack(side="left", padx=10)
        self.thinking = tk.Label(frame1, text="", font=("Arial", 12), fg="gray")
        self.thinking.pack(side="left", padx=10)

        frame2 = tk.Frame(root_win)
        frame2.pack()
        self.blue_val = tk.Label(frame2, text=f"{self._player_label(0)}: --", font=("Arial", 11), fg="blue")
        self.blue_val.pack(side="left", padx=15)
        self.red_val = tk.Label(frame2, text=f"{self._player_label(1)}:  --", font=("Arial", 11), fg="red")
        self.red_val.pack(side="left", padx=15)

        self.draw_board()
        self._start_thinking()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _player_label(self, player):
        name = "Blue" if player == 0 else "Red"
        if not self.vs_random:
            return name
        tag = "MCTS" if player == self.mcts_player else "Random"
        return f"{name} ({tag})"

    # ── background search ────────────────────────────────────────────────────

    def _start_thinking(self):
        winner = self.env.check_win()
        if winner is not None:
            if winner == 2:
                result = "Draw (repetition)"
            elif self.vs_random:
                result = f"{self._player_label(winner)} wins!"
            else:
                result = "Blue wins" if winner == 0 else "Red wins"
            self.status.config(text=f"Done — {result}  ({self.move_num} moves)")
            self.thinking.config(text="")
            return

        player = self.env.player_turn

        # Random player's turn — no search needed
        if self.vs_random and player == self.random_player:
            self.status.config(text=f"Move {self.move_num + 1}  |  {self._player_label(player)} moving...")
            self.thinking.config(text="")
            self.root_win.after(PAUSE_AFTER_MOVE_MS, self._apply_random_move)
            return

        # MCTS player's turn — rebuild root if we don't have one (e.g. after a random move)
        if self.root_node is None:
            prior, _ = _get_prior(self.env, self.net, self.action_dim)
            self.root_node = MCTSNode(prior, self.env.player_turn)

        self.status.config(text=f"Move {self.move_num + 1}  |  {self._player_label(player)} thinking...")
        self.thinking.config(text=f"(0/{self.n_simulations} sims)")

        t = threading.Thread(target=self._search_thread, daemon=True)
        t.start()

    def _search_thread(self):
        if self.leaf_batch_size > 1:
            run_mcts_batched(self.env, self.root_node, self.net,
                             self.n_simulations, self.leaf_batch_size)
        else:
            run_mcts(self.env, self.root_node, self.net, self.n_simulations)
        self.root_win.after(0, self._apply_move)

    # ── apply result on main thread ──────────────────────────────────────────

    def _apply_move(self):
        """Apply the MCTS player's chosen move."""
        player = self.env.player_turn
        canonical_action = int(np.argmax(self.root_node.N))
        real_action = self.env.convert_canonical_action(canonical_action)

        # Value at root = mean backed-up value from current player's perspective
        total_n = self.root_node.N.sum()
        if total_n > 0:
            self.values[player] = float(self.root_node.W.sum() / total_n)

        self.env.agent_action_function(real_action)
        self.move_num += 1

        # Advance tree; will be rebuilt from scratch after the random player moves
        self.root_node = self.root_node.children.get(canonical_action)

        self.status.config(text=f"Move {self.move_num}  |  {self._player_label(player)} moved")
        self.thinking.config(text="")
        self.blue_val.config(text=f"{self._player_label(0)}: {self.values[0]:+.2f}")
        self.red_val.config(text=f"{self._player_label(1)}:  {self.values[1]:+.2f}")
        self.draw_board()
        self.root_win.after(PAUSE_AFTER_MOVE_MS, self._start_thinking)

    def _apply_random_move(self):
        """Apply a uniformly random valid move for the random player."""
        player = self.env.player_turn
        mask = self.env.return_valid_actions_RL()
        valid_indices = np.where(mask > 0)[0]
        real_action = int(np.random.choice(valid_indices))

        self.env.agent_action_function(real_action)
        self.move_num += 1

        # Invalidate the MCTS tree — it was rooted before the random move
        self.root_node = None

        self.status.config(text=f"Move {self.move_num}  |  {self._player_label(player)} moved")
        self.thinking.config(text="")
        self.draw_board()
        self.root_win.after(PAUSE_AFTER_MOVE_MS, self._start_thinking)

    # ── drawing ──────────────────────────────────────────────────────────────

    def draw_board(self):
        self.canvas.delete("all")
        n = self.n
        env = self.env
        cs = CELL_SIZE

        for i in range(n):
            for j in range(n):
                x0, y0 = j * cs, i * cs
                self.canvas.create_rectangle(x0, y0, x0 + cs, y0 + cs, fill="white", outline="gray")

        for j in range(n):
            self.canvas.create_rectangle(j*cs+2, (n-1)*cs+2, j*cs+cs-2, n*cs-2, fill="#d0e8ff", outline="")
            self.canvas.create_rectangle(j*cs+2, 2, j*cs+cs-2, cs-2, fill="#ffd0d0", outline="")

        for row in range(n - 1):
            for col in range(n):
                if env.horizontal_barricades[row][col] != 0:
                    x0, y0 = col * cs, (row + 1) * cs - 3
                    self.canvas.create_rectangle(x0, y0, x0 + cs, y0 + 6, fill="black", outline="")

        for row in range(n):
            for col in range(n - 1):
                if env.vertical_barricades[row][col] != 0:
                    x0, y0 = (col + 1) * cs - 3, row * cs
                    self.canvas.create_rectangle(x0, y0, x0 + 6, y0 + cs, fill="black", outline="")

        for loc, color in [(env.p1loc, "blue"), (env.p2loc, "red")]:
            x0 = loc[1] * cs + 10
            y0 = loc[0] * cs + 10
            self.canvas.create_oval(x0, y0, x0 + cs - 20, y0 + cs - 20,
                                    fill=color, outline="white", width=2)


if __name__ == "__main__":
    VS_RANDOM   = True   # True = MCTS vs Random | False = MCTS vs MCTS (self-play)
    MCTS_PLAYER = 1      # 0 = MCTS plays Blue, 1 = MCTS plays Red (only used when VS_RANDOM=True)

    net, N, BARRIER_COUNT = load_net()

    root = tk.Tk()
    if VS_RANDOM:
        side = "Blue" if MCTS_PLAYER == 0 else "Red"
        mode_str = f"MCTS ({side}) vs Random"
    else:
        mode_str = "Self-play"
    root.title(f"MCTS Live ({N}x{N}) — {mode_str}")
    LiveReplayGUI(root, net, N, BARRIER_COUNT, N_SIMULATIONS,
                  vs_random=VS_RANDOM, mcts_player=MCTS_PLAYER)
    root.mainloop()
