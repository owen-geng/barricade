import tkinter as tk
from tkinter import messagebox
import numpy as np

from environment import Environment

class BarricadeGUI:
    def __init__(self, master, n=7):
        self.master = master
        self.n = n
        self.env = Environment(n=n)
        self.cell_size = 60
        self.selected_action = None  # "move", "hbar", "vbar"

        self.canvas = tk.Canvas(master, width=n*self.cell_size, height=n*self.cell_size, bg="white")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_click)

        # Action buttons
        frame = tk.Frame(master)
        frame.pack()
        tk.Button(frame, text="Move", command=lambda: self.set_action("move")).pack(side="left")
        tk.Button(frame, text="Place Horizontal", command=lambda: self.set_action("hbar")).pack(side="left")
        tk.Button(frame, text="Place Vertical", command=lambda: self.set_action("vbar")).pack(side="left")

        self.draw_board()

    def set_action(self, action):
        self.selected_action = action

    def draw_board(self):
        self.canvas.delete("all")
        # Draw cells
        for i in range(self.n):
            for j in range(self.n):
                x0, y0 = j*self.cell_size, i*self.cell_size
                x1, y1 = x0+self.cell_size, y0+self.cell_size
                self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="black")

        # Draw barricades
        for i in range(self.n-1):
            for j in range(self.n-1):
                if self.env.horizontal_barricades[i][j]:
                    x0, y0 = j*self.cell_size, i*self.cell_size+self.cell_size//2
                    x1, y1 = (j+2)*self.cell_size, i*self.cell_size+self.cell_size//2+5
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="brown", outline="")
                if self.env.vertical_barricades[i][j]:
                    x0, y0 = j*self.cell_size+self.cell_size//2, i*self.cell_size
                    x1, y1 = j*self.cell_size+self.cell_size//2+5, (i+2)*self.cell_size
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="brown", outline="")

        # Draw players
        x0, y0 = self.env.p1loc[1]*self.cell_size+10, self.env.p1loc[0]*self.cell_size+10
        x1, y1 = x0+self.cell_size-20, y0+self.cell_size-20
        self.canvas.create_oval(x0, y0, x1, y1, fill="blue")
        x0, y0 = self.env.p2loc[1]*self.cell_size+10, self.env.p2loc[0]*self.cell_size+10
        x1, y1 = x0+self.cell_size-20, y0+self.cell_size-20
        self.canvas.create_oval(x0, y0, x1, y1, fill="red")

    def on_click(self, event):
        row = event.y // self.cell_size
        col = event.x // self.cell_size

        if self.selected_action == "move":
            self.move_player(row, col)
        elif self.selected_action == "hbar":
            self.place_hbarrier(row, col)
        elif self.selected_action == "vbar":
            self.place_vbarrier(row, col)
        else:
            messagebox.showinfo("Select Action", "Please select an action first!")

        self.draw_board()
        winner = self.env.check_win()
        if winner is not None:
            messagebox.showinfo("Game Over", f"Player {winner+1} wins!")
            self.master.destroy()

    def move_player(self, row, col):
        player = 0 if self.env.player_turn else 1
        valid_moves, _, _ = self.env.return_valid_moves(player)
        move_index = None
        if row < self.env.n and col < self.env.n:
            # Determine direction
            if row == self.env.p1loc[0]-1 and col == self.env.p1loc[1]:
                move_index = 0  # Up
            elif row == self.env.p1loc[0]+1 and col == self.env.p1loc[1]:
                move_index = 2  # Down
            elif row == self.env.p1loc[0] and col == self.env.p1loc[1]+1:
                move_index = 1  # Right
            elif row == self.env.p1loc[0] and col == self.env.p1loc[1]-1:
                move_index = 3  # Left
        if move_index is not None and valid_moves[move_index]:
            self.env.move(player, [row, col])
            self.env.player_turn = not self.env.player_turn

    def place_hbarrier(self, row, col):
        player = 0 if self.env.player_turn else 1
        _, hlist, _ = self.env.return_valid_moves(player)
        index = row*(self.n-1)+col
        if 0 <= row < self.n-1 and 0 <= col < self.n-1 and hlist[index]:
            self.env.place_horizontal_barrier([row, col])
            self.env.player_turn = not self.env.player_turn

    def place_vbarrier(self, row, col):
        player = 0 if self.env.player_turn else 1
        _, _, vlist = self.env.return_valid_moves(player)
        index = row*(self.n-1)+col
        if 0 <= row < self.n-1 and 0 <= col < self.n-1 and vlist[index]:
            self.env.place_vertical_barrier([row, col])
            self.env.player_turn = not self.env.player_turn


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Barricade.gg GUI")
    gui = BarricadeGUI(root, n=7)
    root.mainloop()