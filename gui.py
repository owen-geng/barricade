import tkinter as tk
from tkinter import messagebox
import numpy as np
from environment import Environment


class BarricadeGUI:
    def __init__(self, root, n=7, barricade_count = 10, startpos = None):
        self.n = n
        self.env = Environment(n, barricade_count, startpos)
        self.root = root
        self.cell_size = 60
        self.selected_action = None

        self.valid_moves = None
        self.valid_hbar = None
        self.valid_vbar = None

        self.valid_barricades = None
        

        self.canvas = tk.Canvas(root, width = n*self.cell_size, height = n*self.cell_size, bg = "white")
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.left_click)
        self.root.bind("m", lambda event:self.set_action("Move"))

        frame = tk.Frame(root)
        frame.pack()
        tk.Button(frame, text = "Move", command=lambda: self.set_action("Move")).pack(side="left")
        tk.Button(frame, text = "hbar", command=lambda: self.set_action("hbar")).pack(side="left")
        tk.Button(frame, text = "vbar", command=lambda: self.set_action("vbar")).pack(side="left")
        tk.Button(frame, text = "get representation", command=self.gui_representation_access).pack(side="left")
        tk.Button(frame, text = "get representation", command=self.gui_action_mask_access).pack(side="left")
        self.label = tk.Label(frame, text="Turn: Red" if self.env.player_turn else "Turn: Blue")
        self.label_bluebarricade = tk.Label(frame, text=f"Blue Barricades Remaining: {self.env.barricade_counts[0]}")
        self.label_redbarricade = tk.Label(frame, text=f"Red Barricades Remaining: {self.env.barricade_counts[1]}")
        self.label.pack(side="left")
        self.label_bluebarricade.pack(side = "left")
        self.label_redbarricade.pack(side = "left")


        self.draw_board()
    
    def gui_representation_access(self):
        print(self.env.return_state_representation())

    def gui_action_mask_access(self):
        print(self.env.return_action_mask())

    def debug_func(self):
        #Debugging function
        pass


    def set_action(self, action):
        self.selected_action = action
        print(f"Setting action: {action}")
        if self.selected_action == "Move":
            self.valid_moves = self.env.return_valid_moves(self.env.player_turn)
        else:
            self.valid_moves = None
        
        if self.selected_action == "hbar":
            _, self.valid_hbar, _ = self.env.return_valid_actions(self.env.player_turn)
        else:
            self.valid_hbar = None
        
        if self.selected_action == "vbar":
            _, _, self.valid_vbar = self.env.return_valid_actions(self.env.player_turn)
        else:
            self.valid_vbar = None
        

        self.draw_board() #Redraw the board
    
    def left_click(self, event):
        row = event.y // self.cell_size
        col = event.x // self.cell_size

        if self.selected_action == "Move":
            if self.valid_moves[row][col]:
                self.env.move([row,col])
                print("Moving")
                self.valid_moves = None
                self.selected_action = None
                if self.env.check_win() is not None:
                    if self.env.check_win() == 0:
                        print("Victory for Blue!")
                    else:
                        print("Victory for Red!")

            else:
                print("Invalid Move")
                self.set_action(None)
        elif self.selected_action == "hbar": #Place horizontal barricade
            if self.valid_hbar[row][col]:
                self.env.place_horizontal_barrier([row,col])
                self.set_action(None)
            else:
                print("Invalid hbar")
                self.set_action(None)
        elif self.selected_action == "vbar": #Place horizontal barricade
            if self.valid_vbar[row][col]:
                self.env.place_vertical_barrier([row,col])
                self.set_action(None)
            else:
                print("Invalid vbar")
                self.set_action(None)



        self.draw_board() #Redraw the board

    def draw_board(self):
        self.label.config(text="Turn: Red" if self.env.player_turn else "Turn: Blue")
        self.label_bluebarricade.config(text=f"Blue Barricades Remaining: {self.env.barricade_counts[0]}")
        self.label_redbarricade.config(text=f"Red Barricades Remaining: {self.env.barricade_counts[1]}")
        self.canvas.delete("all")
        # Draw cells
        
        for i in range(self.n):
            for j in range(self.n):
                x0, y0 = j*self.cell_size, i*self.cell_size
                x1, y1 = x0+self.cell_size, y0+self.cell_size

                if self.selected_action == "Move":
                    if self.valid_moves[i][j]:
                        self.canvas.create_rectangle(x0, y0, x1, y1, fill="green", outline="gray")
                    else:
                        self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="gray")
                else:
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="white", outline="gray")


        # Draw barricades
        #Horizontal barricades
        for row in range(self.n-1):
            for col in range(self.n):
                if self.env.horizontal_barricades[row][col] != 0:
                    x0, y0 = col*self.cell_size, (row+1)*self.cell_size-2.5
                    x1, y1 = (col+1)*self.cell_size, (row+1)*self.cell_size+2.5
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="black", outline="")
        
        #Vertical barricades
        for row in range(self.n):
            for col in range(self.n-1):
                if self.env.vertical_barricades[row][col] != 0:
                    x0, y0 = (col+1)*self.cell_size-2.5, (row)*self.cell_size
                    x1, y1 = (col+1)*self.cell_size+2.5, (row+1)*self.cell_size
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="black", outline="")

        # Draw players
        x0, y0 = self.env.p1loc[1]*self.cell_size+10, self.env.p1loc[0]*self.cell_size+10
        x1, y1 = x0+self.cell_size-20, y0+self.cell_size-20
        self.canvas.create_oval(x0, y0, x1, y1, fill="blue")
        x0, y0 = self.env.p2loc[1]*self.cell_size+10, self.env.p2loc[0]*self.cell_size+10
        x1, y1 = x0+self.cell_size-20, y0+self.cell_size-20
        self.canvas.create_oval(x0, y0, x1, y1, fill="red")


def test_button(str = "Testing!"):
    print(str)

root = tk.Tk()
root.title("Barricade game")

gui = BarricadeGUI(root, n=7)
root.mainloop()