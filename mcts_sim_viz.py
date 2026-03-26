"""
mcts_sim_viz.py — watch MCTS think in real time.

During each search, the board shows the position being explored at each
simulation step.  After all simulations, the best move is applied and the
next search begins.

Two canvases are drawn side by side:
  Left  — the real game position (updated only when a move is played)
  Right — the position being simulated (streams live during search)
"""
import os
import queue
import threading
import tkinter as tk
import numpy as np
import torch
from environment import Environment
from MCTS_agent import (AlphaZeroNet, MCTSNode, _get_prior,
                        select_action, _backup, device)

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

# ── config ────────────────────────────────────────────────────────────────────
N             = 9
BARRIER_COUNT = 10
N_SIMULATIONS = 100
CELL_SIZE     = 80
POLL_MS       = 16      # ~60 fps drain / redraw
MOVE_PAUSE_MS = 800     # pause between moves so you can see the result


# ── modified MCTS that streams board states ───────────────────────────────────

def run_mcts_viz(root_env: Environment, root_node: MCTSNode, net,
                 n_simulations: int, step_q: queue.Queue, c_puct: float = 1.5):
    """Like run_mcts but posts each board position to step_q."""
    action_dim = len(root_node.P)

    for sim in range(n_simulations):
        env = root_env.clone()
        node = root_node
        path = []

        while True:
            action = select_action(node, c_puct)
            path.append((node, action))
            env.agent_action_function(env.convert_canonical_action(action))

            step_q.put({
                'type':   'step',
                'sim':    sim,
                'total':  n_simulations,
                'depth':  len(path),
                'p1loc':  env.p1loc.copy(),
                'p2loc':  env.p2loc.copy(),
                'h_bar':  env.horizontal_barricades.copy(),
                'v_bar':  env.vertical_barricades.copy(),
            })

            winner = env.check_win()
            if winner is not None:
                val = 0.0 if winner == 2 else (1.0 if winner == path[-1][0].player else -1.0)
                _backup(path, val)
                break

            if action not in node.children:
                prior, value = _get_prior(env, net, action_dim)
                node.children[action] = MCTSNode(prior, env.player_turn)
                _backup(path, -value)
                break

            node = node.children[action]

    step_q.put({'type': 'done'})


# ── GUI ───────────────────────────────────────────────────────────────────────

class SimVizGUI:
    def __init__(self, root_win, net, n, barrier_count, n_simulations):
        self.root_win     = root_win
        self.net          = net
        self.n            = n
        self.n_simulations = n_simulations
        self.action_dim   = 12 + 2 * (n - 1) ** 2
        self.step_q       = queue.Queue()   # unbounded — sim runs at full speed
        self.move_num     = 0
        self._polling     = False

        self.env = Environment(n, barrier_count)
        prior, _ = _get_prior(self.env, net, self.action_dim)
        self.root_node = MCTSNode(prior, self.env.player_turn)

        cs = CELL_SIZE
        board_w = n * cs

        # ── layout ────────────────────────────────────────────────────────────
        canvas_frame = tk.Frame(root_win)
        canvas_frame.pack()

        left_frame = tk.Frame(canvas_frame)
        left_frame.grid(row=0, column=0, padx=10)
        tk.Label(left_frame, text='Game position', font=('Arial', 11, 'bold')).pack()
        self.game_canvas = tk.Canvas(left_frame, width=board_w, height=board_w, bg='white')
        self.game_canvas.pack()

        right_frame = tk.Frame(canvas_frame)
        right_frame.grid(row=0, column=1, padx=10)
        tk.Label(right_frame, text='Simulation view', font=('Arial', 11, 'bold')).pack()
        self.sim_canvas = tk.Canvas(right_frame, width=board_w, height=board_w, bg='white')
        self.sim_canvas.pack()

        # ── status bar ────────────────────────────────────────────────────────
        status_frame = tk.Frame(root_win)
        status_frame.pack(pady=4)
        self.move_label = tk.Label(status_frame, text='', font=('Arial', 12))
        self.move_label.pack(side='left', padx=12)
        self.sim_label  = tk.Label(status_frame, text='', font=('Arial', 11), fg='gray')
        self.sim_label.pack(side='left', padx=12)

        # Draw initial real board and start first search
        self._draw_real_board()
        self._start_search()

    # ── search lifecycle ──────────────────────────────────────────────────────

    def _start_search(self):
        winner = self.env.check_win()
        if winner is not None:
            result = ('Blue wins' if winner == 0
                      else 'Red wins' if winner == 1
                      else 'Draw (repetition)')
            self.move_label.config(text=f'Done — {result}  ({self.move_num} moves)')
            self.sim_label.config(text='')
            return

        player = self.env.player_turn
        self.move_label.config(
            text=f'Move {self.move_num + 1}  |  {"Blue" if player == 0 else "Red"} thinking...')
        self.sim_label.config(text='(starting...)')

        t = threading.Thread(
            target=run_mcts_viz,
            args=(self.env, self.root_node, self.net, self.n_simulations, self.step_q),
            daemon=True,
        )
        t.start()
        if not self._polling:
            self._polling = True
            self.root_win.after(POLL_MS, self._poll)

    def _poll(self):
        """Drain queue, display latest simulated state; on 'done' apply move."""
        latest = None
        done   = False
        try:
            while True:
                item = self.step_q.get_nowait()
                if item['type'] == 'done':
                    done = True
                    break
                latest = item
        except queue.Empty:
            pass

        if latest is not None:
            self.sim_label.config(
                text=f'sim {latest["sim"]+1}/{latest["total"]}  depth {latest["depth"]}')
            self._draw_canvas(self.sim_canvas,
                              latest['p1loc'], latest['p2loc'],
                              latest['h_bar'],  latest['v_bar'])

        if done:
            self._polling = False
            self._apply_move()
        else:
            self.root_win.after(POLL_MS, self._poll)

    # ── move application ─────────────────────────────────────────────────────

    def _apply_move(self):
        player = self.env.player_turn
        canonical_action = int(np.argmax(self.root_node.N))
        real_action      = self.env.convert_canonical_action(canonical_action)

        self.env.agent_action_function(real_action)
        self.move_num += 1

        self.root_node = self.root_node.children.get(canonical_action)
        if self.root_node is None:
            prior, _ = _get_prior(self.env, self.net, self.action_dim)
            self.root_node = MCTSNode(prior, self.env.player_turn)

        name = 'Blue' if player == 0 else 'Red'
        self.move_label.config(text=f'Move {self.move_num}  |  {name} moved')
        self.sim_label.config(text='')

        # Update both canvases to the real position
        self._draw_real_board()
        self._draw_canvas(self.sim_canvas,
                          self.env.p1loc, self.env.p2loc,
                          self.env.horizontal_barricades,
                          self.env.vertical_barricades)

        self.root_win.after(MOVE_PAUSE_MS, self._start_search)

    # ── drawing ───────────────────────────────────────────────────────────────

    def _draw_real_board(self):
        self._draw_canvas(self.game_canvas,
                          self.env.p1loc, self.env.p2loc,
                          self.env.horizontal_barricades,
                          self.env.vertical_barricades)

    def _draw_canvas(self, canvas, p1loc, p2loc, h_bar, v_bar):
        canvas.delete('all')
        n, cs = self.n, CELL_SIZE

        # cells
        for i in range(n):
            for j in range(n):
                canvas.create_rectangle(
                    j*cs, i*cs, (j+1)*cs, (i+1)*cs, fill='white', outline='gray')

        # goal rows (blue bottom, red top — matches mcts_replay colouring)
        for j in range(n):
            canvas.create_rectangle(
                j*cs+2, (n-1)*cs+2, (j+1)*cs-2, n*cs-2, fill='#d0e8ff', outline='')
            canvas.create_rectangle(
                j*cs+2, 2, (j+1)*cs-2, cs-2, fill='#ffd0d0', outline='')

        # horizontal barricades
        for row in range(n - 1):
            for col in range(n):
                if h_bar[row][col] != 0:
                    x0, y0 = col*cs, (row+1)*cs - 3
                    canvas.create_rectangle(x0, y0, x0+cs, y0+6, fill='black', outline='')

        # vertical barricades
        for row in range(n):
            for col in range(n - 1):
                if v_bar[row][col] != 0:
                    x0, y0 = (col+1)*cs - 3, row*cs
                    canvas.create_rectangle(x0, y0, x0+6, y0+cs, fill='black', outline='')

        # pieces
        for loc, color in [(p1loc, 'blue'), (p2loc, 'red')]:
            x0 = loc[1]*cs + 10
            y0 = loc[0]*cs + 10
            canvas.create_oval(
                x0, y0, x0+cs-20, y0+cs-20, fill=color, outline='white', width=2)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    net, N, BARRIER_COUNT = load_net()

    root = tk.Tk()
    root.title(f'MCTS Simulation Viewer ({N}x{N})')
    SimVizGUI(root, net, N, BARRIER_COUNT, N_SIMULATIONS)
    root.mainloop()
