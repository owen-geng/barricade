#Environment for barricade.gg game
import numpy as np
from collections import deque

class Environment:
    def __init__(self, n = 7, barricade_count = 10, startpos = None):
        if startpos is None:
            startpos = int(np.floor(n/2))

        self.n = n
        
        self.board = np.zeros([n, n])

        self.p1loc = [0, startpos]
        self.p2loc = [n-1, startpos]

        self.barricade_counts = [barricade_count, barricade_count]
        self.horizontal_barricades = np.zeros([n,n])
        self.vertical_barricades = np.zeros([n,n])
        self.turn_count = 0
        self.player_turn = 0
        self.barricade_id = 1
    

    def blocked_paths(self, loc): #Loc is a 2x2 arr
        valid = [False, False, False, False] #Up, Right, Down, Left
        if loc[0] != 0: #If row is NOT 0, i.e, NOT the top
            if self.horizontal_barricades[loc[0]-1][loc[1]] == 0:
                valid[0] = True 
        if loc[0] != self.n-1: #If row is NOT n-1, i.e, NOT the bottom
            if self.horizontal_barricades[loc[0]][loc[1]] == 0:
                valid[2] = True
        if loc[1] != 0: #If column is NOT 0, i.e, NOT the left
            if self.vertical_barricades[loc[0]][loc[1]-1] == 0:
                valid[3] = True
        if loc[1] != self.n-1: #If column is NOT n-1, i.e, NOT the right
            if self.vertical_barricades[loc[0]][loc[1]] == 0:
                valid[1] = True    
        return valid

    def possible_path(self, player, loc = None):
        if loc is None:
            start = self.p1loc if player == 0 else self.p2loc
        else:
            start = loc
        
        target = self.n - 1 if player == 0 else 0

        stack = deque([start])
        visited = np.zeros((self.n, self.n), dtype=bool)
        visited[start[0]][start[1]] = True

        while stack:
            row, col = stack.popleft()

            if row == target:
                return True

            can_up, can_right, can_down, can_left = self.blocked_paths([row, col])

            if can_up and row > 0 and not visited[row-1][col]:
                visited[row-1][col] = True
                stack.append((row-1, col))

            if can_right and col < self.n-1 and not visited[row][col+1]:
                visited[row][col+1] = True
                stack.append((row, col+1))

            if can_down and row < self.n-1 and not visited[row+1][col]:
                visited[row+1][col] = True
                stack.append((row+1, col))

            if can_left and col > 0 and not visited[row][col-1]:
                visited[row][col-1] = True
                stack.append((row, col-1))

        return False
    
    def return_valid_moves(self, player):
        loc = self.p1loc if player == 0 else self.p2loc
        otherloc = self.p2loc if player == 0 else self.p1loc

        moves = np.zeros((self.n, self.n))

        valid = self.blocked_paths(loc)

        directions = [[-1,0],[0,1],[1,0],[0,-1]] #Directions
        for ind, direct in enumerate(directions):
            if valid[ind]:
                newloc = [loc[0]+direct[0], loc[1]+direct[1]]
                if newloc == otherloc: #Jump behaviour
                    jumpvalid = self.blocked_paths(newloc)
                    #If not blocked on the other side
                    if jumpvalid[ind]: #Not blocked
                        newloc = [loc[0]+(direct[0]*2), loc[1]+(direct[1]*2)]
                        moves[newloc[0]][newloc[1]] = 1
                    else: #Blocked, diagonals
                        #Jumping along horizontal, check verticals
                        if ind == 1 or ind == 3:
                            if jumpvalid[0]:
                                moves[loc[0]+direct[0]+directions[0][0]][loc[1]+direct[1]+directions[0][1]] = 1
                            if jumpvalid[2]:
                                moves[loc[0]+direct[0]+directions[2][0]][loc[1]+direct[1]+directions[2][1]] = 1
                        #Jumping along verticals, check horizontals
                        if ind == 0 or ind == 2:
                            if jumpvalid[1]:
                                moves[loc[0]+direct[0]+directions[1][0]][loc[1]+direct[1]+directions[1][1]] = 1
                            if jumpvalid[3]:
                                moves[loc[0]+direct[0]+directions[3][0]][loc[1]+direct[1]+directions[3][1]] = 1

                else:
                    moves[newloc[0]][newloc[1]] = 1
        
        return moves


    
    def return_valid_actions(self, player):
        #Valid moves representation. First n by n represent all visitable squares. Next n-1 by n-1 represent h_barriers, and next n-1 by n-1 represent v_barriers.
        #By convention, placing a hbarrier in [0, 0] places one in [0,1] as well, and placing a vbarrier in [0,0] places one in [1,0] as well.
        moves = self.return_valid_moves(player)
        horizontal_list = np.zeros((self.n, self.n))
        vertical_list = np.zeros((self.n, self.n))

        if self.barricade_counts[player] <= 0:
            return moves, horizontal_list, vertical_list
        
        for row in range(self.n-1):
            for col in range(self.n-1):
                #Check horizontal
                if self.horizontal_barricades[row][col] == 0 and self.horizontal_barricades[row][col+1] == 0 and ((self.vertical_barricades[row][col] != self.vertical_barricades[row+1][col]) or self.vertical_barricades[row][col] == 0):
                    #If no overlap and no cross

                    #Check if paths are still possible
                    self.horizontal_barricades[row][col] = self.barricade_id
                    self.horizontal_barricades[row][col+1] = self.barricade_id
                    if self.possible_path(0) and self.possible_path(1): #If both can still make it
                        horizontal_list[row][col] = 1
                    else:
                        pass
                    
                    self.horizontal_barricades[row][col] = 0
                    self.horizontal_barricades[row][col+1] = 0
                else:
                    pass
                
                #Check vertical
                if self.vertical_barricades[row][col] == 0 and self.vertical_barricades[row+1][col] == 0 and ((self.horizontal_barricades[row][col] != self.horizontal_barricades[row][col+1]) or self.horizontal_barricades[row][col] == 0):

                    self.vertical_barricades[row][col] = self.barricade_id
                    self.vertical_barricades[row+1][col] = self.barricade_id
                    if self.possible_path(0) and self.possible_path(1): #If both can still make it
                        vertical_list[row][col] = 1
                    else:
                        pass
                    self.vertical_barricades[row][col] = 0
                    self.vertical_barricades[row+1][col] = 0
                else:
                    pass
        
        return moves, horizontal_list, vertical_list


    def move(self, loc): #Unsafe
        if self.player_turn == 0:
            player = 0
        else:
            player = 1
        
        if player == 0:
            self.p1loc = loc
        elif player == 1:
            self.p2loc = loc
        else:
            print("Something went wrong in move")
            return -1

        self.player_turn = 0 if self.player_turn == 1 else 1

        self.turn_count += 1 #Increment turn count
        return 0
        

    
    
    def place_horizontal_barrier(self, loc):

        if loc[0] > self.n-1 or loc[1] > self.n-1:
            print("Unsafe hbarrier!")
            return -1

        self.horizontal_barricades[loc[0]][loc[1]] = self.barricade_id
        self.horizontal_barricades[loc[0]][loc[1]+1] = self.barricade_id
        self.barricade_id += 1

        self.barricade_counts[self.player_turn] -= 1

        self.player_turn = 0 if self.player_turn == 1 else 1
        return 0
    
    def place_vertical_barrier(self, loc):
        if loc[0] > self.n-1 or loc[1] > self.n-1:
            print("Unsafe vbarrier!")
            return -1
        self.vertical_barricades[loc[0]][loc[1]] = self.barricade_id
        self.vertical_barricades[loc[0]+1][loc[1]] = self.barricade_id
        self.barricade_id += 1
        self.barricade_counts[self.player_turn] -= 1

        self.player_turn = 0 if self.player_turn == 1 else 1
        return 0

    def check_win(self):
        if self.p1loc[0] == self.n-1:
            return 0
        if self.p2loc[0] == 0:
            return 1
        return None
    

    
    
    def move_debug(self, player, loc): #Unsafe! For debugging

        if player == 0:
            self.p1loc = loc
        elif player == 1:
            self.p2loc = loc
        else:
            return -1
        return 0
    