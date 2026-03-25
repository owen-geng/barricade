import tkinter as tk
import torch
from environment import Environment
from DQN_agent import AgentNet

CELL_SIZE = 60
STEP_DELAY_MS = 400  # ms between moves
NUM_GAMES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_agent(path="dqn.pt"):
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        sd = ckpt['state_dict']
        n = ckpt['n']
        barrier_count = ckpt['barrier_count']
    else:
        # Legacy checkpoint: bare state_dict, infer n from output dim
        sd = {k.replace("_orig_mod.", ""): v for k, v in ckpt.items()}
        output_dim = sd["head.2.bias"].shape[0]
        n = int((((output_dim - 12) / 2) ** 0.5) + 1)
        barrier_count = 10  # old default
    net = AgentNet(n, 12 + 2*(n-1)*(n-1)).to(device)
    net.load_state_dict(sd)
    net.eval()
    return net, n, barrier_count

def greedy_action(net, env):
    state = env.return_state_representation().unsqueeze(0).to(device)
    mask = env.return_action_mask()
    with torch.no_grad():
        q = net(state)[0]
        q[~mask.to(device)] = -float("inf")
    return q.argmax().item()


class ReplayGUI:
    def __init__(self, root, net, n, barrier_count):
        self.root = root
        self.net = net
        self.n = n
        self.barrier_count = barrier_count
        self.game_num = 0
        self.env = Environment(n, barrier_count)
        self.cell_size = CELL_SIZE

        self.canvas = tk.Canvas(root, width=n*CELL_SIZE, height=n*CELL_SIZE, bg="white")
        self.canvas.pack()

        frame = tk.Frame(root)
        frame.pack()
        self.status = tk.Label(frame, text="", font=("Arial", 12))
        self.status.pack(side="left", padx=10)
        self.turn_label = tk.Label(frame, text="", font=("Arial", 12))
        self.turn_label.pack(side="left", padx=10)

        self.start_game()

    def start_game(self):
        self.game_num += 1
        self.env.__init__(self.n, self.barrier_count)
        self.turn = 0
        self.status.config(text=f"Game {self.game_num}/{NUM_GAMES}")
        self.draw_board()
        self.root.after(STEP_DELAY_MS, self.step)

    def step(self):
        winner = self.env.check_win()
        if winner is not None or self.turn >= 200:
            result = f"Game {self.game_num}: {'Blue wins' if winner == 0 else 'Red wins' if winner == 1 else 'Draw (timeout)'}"
            self.status.config(text=result)
            if self.game_num < NUM_GAMES:
                self.root.after(1500, self.start_game)
            else:
                self.status.config(text="All games finished.")
            return

        action = greedy_action(self.net, self.env)
        p = self.env.player_turn
        self.env.agent_action_function(action)
        self.turn += 1
        self.turn_label.config(text=f"Turn {self.turn}  |  {'Blue' if p == 0 else 'Red'} moves")
        self.draw_board()
        self.root.after(STEP_DELAY_MS, self.step)

    def draw_board(self):
        self.canvas.delete("all")
        n = self.n
        env = self.env
        cs = self.cell_size

        # Cells
        for i in range(n):
            for j in range(n):
                x0, y0 = j*cs, i*cs
                self.canvas.create_rectangle(x0, y0, x0+cs, y0+cs, fill="white", outline="gray")

        # Goal rows (blue = row n-1, red = row 0)
        for j in range(n):
            self.canvas.create_rectangle(j*cs+2, (n-1)*cs+2, j*cs+cs-2, n*cs-2, fill="#d0e8ff", outline="")
            self.canvas.create_rectangle(j*cs+2, 2, j*cs+cs-2, cs-2, fill="#ffd0d0", outline="")

        # Horizontal barricades
        for row in range(n-1):
            for col in range(n):
                if env.horizontal_barricades[row][col] != 0:
                    x0, y0 = col*cs, (row+1)*cs - 3
                    self.canvas.create_rectangle(x0, y0, x0+cs, y0+6, fill="black", outline="")

        # Vertical barricades
        for row in range(n):
            for col in range(n-1):
                if env.vertical_barricades[row][col] != 0:
                    x0, y0 = (col+1)*cs - 3, row*cs
                    self.canvas.create_rectangle(x0, y0, x0+6, y0+cs, fill="black", outline="")

        # Players
        for loc, color in [(env.p1loc, "blue"), (env.p2loc, "red")]:
            x0 = loc[1]*cs + 10
            y0 = loc[0]*cs + 10
            self.canvas.create_oval(x0, y0, x0+cs-20, y0+cs-20, fill=color, outline="white", width=2)


net, n, barrier_count = load_agent("dqn.pt")
root = tk.Tk()
root.title(f"DQN Agent — Self-Play ({n}x{n})")
app = ReplayGUI(root, net, n, barrier_count)
root.mainloop()
