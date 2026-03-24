#Environment for barricade.gg game
import numpy as np

import torch

try:
    from numba import njit as _njit
    _NUMBA = True
except ImportError:
    _NUMBA = False
    def _njit(*args, **kwargs):
        def decorator(f): return f
        return decorator if args and callable(args[0]) else (lambda f: f) if not args else decorator


@_njit(cache=True)
def _jit_possible_path(h_bars, v_bars, n, start_r, start_c, target_row):
    """BFS reachability check compiled to machine code by Numba.
    Uses a fixed-size array queue to avoid Python deque overhead.
    """
    visited = np.zeros((n, n), dtype=np.bool_)
    queue_r = np.empty(n * n, dtype=np.int32)
    queue_c = np.empty(n * n, dtype=np.int32)
    head = 0; tail = 0
    queue_r[0] = start_r; queue_c[0] = start_c; tail = 1
    visited[start_r, start_c] = True

    while head < tail:
        r = queue_r[head]; c = queue_c[head]; head += 1
        if r == target_row:
            return True
        if r > 0 and h_bars[r-1, c] == 0 and not visited[r-1, c]:
            visited[r-1, c] = True
            queue_r[tail] = r-1; queue_c[tail] = c; tail += 1
        if c < n-1 and v_bars[r, c] == 0 and not visited[r, c+1]:
            visited[r, c+1] = True
            queue_r[tail] = r; queue_c[tail] = c+1; tail += 1
        if r < n-1 and h_bars[r, c] == 0 and not visited[r+1, c]:
            visited[r+1, c] = True
            queue_r[tail] = r+1; queue_c[tail] = c; tail += 1
        if c > 0 and v_bars[r, c-1] == 0 and not visited[r, c-1]:
            visited[r, c-1] = True
            queue_r[tail] = r; queue_c[tail] = c-1; tail += 1
    return False


@_njit(cache=True)
def _jit_path_length(h_bars, v_bars, n, start_r, start_c, target_row):
    """BFS shortest-path distance to goal row. Returns n*n if unreachable."""
    visited = np.zeros((n, n), dtype=np.bool_)
    queue_r = np.empty(n * n, dtype=np.int32)
    queue_c = np.empty(n * n, dtype=np.int32)
    queue_d = np.empty(n * n, dtype=np.int32)
    head = 0; tail = 0
    queue_r[0] = start_r; queue_c[0] = start_c; queue_d[0] = 0; tail = 1
    visited[start_r, start_c] = True

    while head < tail:
        r = queue_r[head]; c = queue_c[head]; d = queue_d[head]; head += 1
        if r == target_row:
            return d
        if r > 0 and h_bars[r-1, c] == 0 and not visited[r-1, c]:
            visited[r-1, c] = True
            queue_r[tail] = r-1; queue_c[tail] = c; queue_d[tail] = d+1; tail += 1
        if c < n-1 and v_bars[r, c] == 0 and not visited[r, c+1]:
            visited[r, c+1] = True
            queue_r[tail] = r; queue_c[tail] = c+1; queue_d[tail] = d+1; tail += 1
        if r < n-1 and h_bars[r, c] == 0 and not visited[r+1, c]:
            visited[r+1, c] = True
            queue_r[tail] = r+1; queue_c[tail] = c; queue_d[tail] = d+1; tail += 1
        if c > 0 and v_bars[r, c-1] == 0 and not visited[r, c-1]:
            visited[r, c-1] = True
            queue_r[tail] = r; queue_c[tail] = c-1; queue_d[tail] = d+1; tail += 1
    return n * n


@_njit(cache=True)
def _jit_max_flow(h_bars, v_bars, n, start_r, start_c, target_row, cap):
    """BFS augmenting-path max-flow compiled to machine code by Numba.
    Replaces Python dict/deque with fixed-size numpy arrays.
    net_flow[r,c,d]: net flow from (r,c) in direction d (0=up,1=right,2=down,3=left).
    """
    net_flow = np.zeros((n, n, 4), dtype=np.int8)
    visited  = np.zeros((n, n), dtype=np.bool_)
    prev_r   = np.empty((n, n), dtype=np.int8)
    prev_c   = np.empty((n, n), dtype=np.int8)
    prev_d   = np.empty((n, n), dtype=np.int8)
    queue_r  = np.empty(n * n, dtype=np.int32)
    queue_c  = np.empty(n * n, dtype=np.int32)

    flow = 0
    while flow < cap:
        visited[:, :] = False
        visited[start_r, start_c] = True
        prev_d[start_r, start_c] = -1  # source sentinel
        head = 0; tail = 0
        queue_r[0] = start_r; queue_c[0] = start_c; tail = 1
        sink_r = -1; sink_c = -1

        while head < tail:
            r = queue_r[head]; c = queue_c[head]; head += 1
            if r == target_row:
                sink_r = r; sink_c = c; break
            # up (d=0)
            if r > 0 and h_bars[r-1, c] == 0 and net_flow[r, c, 0] < 1:
                nr = r - 1; nc = c
                if not visited[nr, nc]:
                    visited[nr, nc] = True
                    prev_r[nr, nc] = r; prev_c[nr, nc] = c; prev_d[nr, nc] = 0
                    queue_r[tail] = nr; queue_c[tail] = nc; tail += 1
            # right (d=1)
            if c < n-1 and v_bars[r, c] == 0 and net_flow[r, c, 1] < 1:
                nr = r; nc = c + 1
                if not visited[nr, nc]:
                    visited[nr, nc] = True
                    prev_r[nr, nc] = r; prev_c[nr, nc] = c; prev_d[nr, nc] = 1
                    queue_r[tail] = nr; queue_c[tail] = nc; tail += 1
            # down (d=2)
            if r < n-1 and h_bars[r, c] == 0 and net_flow[r, c, 2] < 1:
                nr = r + 1; nc = c
                if not visited[nr, nc]:
                    visited[nr, nc] = True
                    prev_r[nr, nc] = r; prev_c[nr, nc] = c; prev_d[nr, nc] = 2
                    queue_r[tail] = nr; queue_c[tail] = nc; tail += 1
            # left (d=3)
            if c > 0 and v_bars[r, c-1] == 0 and net_flow[r, c, 3] < 1:
                nr = r; nc = c - 1
                if not visited[nr, nc]:
                    visited[nr, nc] = True
                    prev_r[nr, nc] = r; prev_c[nr, nc] = c; prev_d[nr, nc] = 3
                    queue_r[tail] = nr; queue_c[tail] = nc; tail += 1

        if sink_r == -1:
            break

        # Backtrack and update net flow
        r = sink_r; c = sink_c
        while True:
            d = prev_d[r, c]
            if d == -1:
                break
            pr = prev_r[r, c]; pc = prev_c[r, c]
            net_flow[pr, pc, d] += np.int8(1)
            net_flow[r, c, (d + 2) % 4] -= np.int8(1)
            r = pr; c = pc
        flow += 1

    return flow, net_flow


# Warm up JIT compilation at import time so the first actual call isn't slow.
# With cache=True this only compiles once; subsequent runs load from disk (~0.1s).
if _NUMBA:
    _w_h = np.zeros((7, 7), dtype=np.int64)
    _w_v = np.zeros((7, 7), dtype=np.int64)
    _jit_possible_path(_w_h, _w_v, 7, 0, 3, 6)
    _jit_max_flow(_w_h, _w_v, 7, 0, 3, 6, 3)

# All possible movement destinations expressed as (row, col) offsets from the
# current player's position. Covers regular moves, straight jumps, and all
# diagonal jump variants — 12 fixed offsets regardless of board size.
MOVE_OFFSETS = [
    (-2,  0),  # 0:  jump up (straight over opponent)
    (-1, -1),  # 1:  diagonal up-left
    (-1,  0),  # 2:  up
    (-1, +1),  # 3:  diagonal up-right
    ( 0, -2),  # 4:  jump left (straight over opponent)
    ( 0, -1),  # 5:  left
    ( 0, +1),  # 6:  right
    ( 0, +2),  # 7:  jump right (straight over opponent)
    (+1, -1),  # 8:  diagonal down-left
    (+1,  0),  # 9:  down
    (+1, +1),  # 10: diagonal down-right
    (+2,  0),  # 11: jump down (straight over opponent)
]
N_MOVE_ACTIONS = len(MOVE_OFFSETS)  # 12

class Environment:
    def __init__(self, n = 7, barricade_count = 10, startpos = None):
        if startpos is None:
            startpos = int(np.floor(n/2))

        self.n = n
        
        self.board = np.zeros([n, n])

        self.p1loc = [0, startpos]
        self.p2loc = [n-1, startpos]

        self.barricade_counts = [barricade_count, barricade_count]
        self.horizontal_barricades = np.zeros([n,n], dtype=int)
        self.vertical_barricades = np.zeros([n,n], dtype=int)
        self.turn_count = 0
        self.player_turn = 0
        self.barricade_id = 1
        self._visited = np.zeros((n, n), dtype=bool)  # reused by possible_path
    

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
        start = loc if loc is not None else (self.p1loc if player == 0 else self.p2loc)
        target = self.n - 1 if player == 0 else 0
        return bool(_jit_possible_path(
            self.horizontal_barricades, self.vertical_barricades,
            self.n, start[0], start[1], target
        ))
    
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


    
    def _max_flow_to_goal(self, player, cap=3):
        source = self.p1loc if player == 0 else self.p2loc
        target_row = self.n - 1 if player == 0 else 0
        return _jit_max_flow(
            self.horizontal_barricades, self.vertical_barricades,
            self.n, source[0], source[1], target_row, cap
        )

    def _dangerous_edge_arrays(self, nf0, nf1):
        """Return (dangerous_down, dangerous_right) boolean arrays of shape (n, n).
        dangerous_down[r,c]  = True if edge (r,c)↔(r+1,c)  carries flow.
        dangerous_right[r,c] = True if edge (r,c)↔(r,c+1)  carries flow.
        An edge is dangerous if either player's max-flow uses it in either direction.
        Must use != 0 (not > 0) because flow can route in the non-canonical direction,
        which is stored as a negative net-flow value on the canonical direction entry.
        """
        dangerous_down  = (nf0[:, :, 2] != 0) | (nf1[:, :, 2] != 0)
        dangerous_right = (nf0[:, :, 1] != 0) | (nf1[:, :, 1] != 0)
        return dangerous_down, dangerous_right

    def return_valid_actions(self, player = None):
        if player == None:
            player = self.player_turn
        moves = self.return_valid_moves(player)
        horizontal_list = np.zeros((self.n, self.n))
        vertical_list = np.zeros((self.n, self.n))

        if self.barricade_counts[player] <= 0:
            return moves, horizontal_list, vertical_list

        flow0, nf0 = self._max_flow_to_goal(0)
        flow1, nf1 = self._max_flow_to_goal(1)
        if flow0 < 1 or flow1 < 1:
            return moves, horizontal_list, vertical_list
        all_safe = flow0 >= 3 and flow1 >= 3

        n = self.n
        h = self.horizontal_barricades
        v = self.vertical_barricades

        # Vectorized overlap-validity masks
        h_ok = (h[:n-1, :n-1] == 0) & (h[:n-1, 1:n] == 0) & \
               ((v[:n-1, :n-1] != v[1:n, :n-1]) | (v[:n-1, :n-1] == 0))
        v_ok = (v[:n-1, :n-1] == 0) & (v[1:n, :n-1] == 0) & \
               ((h[:n-1, :n-1] != h[:n-1, 1:n]) | (h[:n-1, :n-1] == 0))

        if all_safe:
            horizontal_list[:n-1, :n-1] = h_ok
            vertical_list[:n-1, :n-1] = v_ok
            return moves, horizontal_list, vertical_list

        ddown, dright = self._dangerous_edge_arrays(nf0, nf1)
        hbar_edge_safe = ~(ddown[:n-1, :n-1] | ddown[:n-1, 1:n])
        vbar_edge_safe = ~(dright[:n-1, :n-1] | dright[1:n, :n-1])

        horizontal_list[:n-1, :n-1] = h_ok & hbar_edge_safe
        vertical_list[:n-1, :n-1] = v_ok & vbar_edge_safe

        for row, col in zip(*np.where(h_ok & ~hbar_edge_safe)):
            self.horizontal_barricades[row][col] = self.barricade_id
            self.horizontal_barricades[row][col+1] = self.barricade_id
            if self.possible_path(0) and self.possible_path(1):
                horizontal_list[row][col] = 1
            self.horizontal_barricades[row][col] = 0
            self.horizontal_barricades[row][col+1] = 0

        for row, col in zip(*np.where(v_ok & ~vbar_edge_safe)):
            self.vertical_barricades[row][col] = self.barricade_id
            self.vertical_barricades[row+1][col] = self.barricade_id
            if self.possible_path(0) and self.possible_path(1):
                vertical_list[row][col] = 1
            self.vertical_barricades[row][col] = 0
            self.vertical_barricades[row+1][col] = 0

        return moves, horizontal_list, vertical_list

    def return_valid_actions_RL(self, player = None):
        if player == None:
            player = self.player_turn
        move_grid = self.return_valid_moves(player)
        loc = self.p1loc if player == 0 else self.p2loc
        moves = np.zeros(N_MOVE_ACTIONS)
        for i, (dr, dc) in enumerate(MOVE_OFFSETS):
            r, c = loc[0] + dr, loc[1] + dc
            if 0 <= r < self.n and 0 <= c < self.n:
                moves[i] = move_grid[r][c]
        horizontal_list = np.zeros((self.n-1, self.n-1))
        vertical_list = np.zeros((self.n-1, self.n-1))

        if self.barricade_counts[player] <= 0:
            return np.concatenate([moves, horizontal_list.flatten(), vertical_list.flatten()])

        flow0, nf0 = self._max_flow_to_goal(0)
        flow1, nf1 = self._max_flow_to_goal(1)
        if flow0 < 1 or flow1 < 1:
            return np.concatenate([moves, horizontal_list.flatten(), vertical_list.flatten()])
        all_safe = flow0 >= 3 and flow1 >= 3

        n = self.n
        h = self.horizontal_barricades
        v = self.vertical_barricades

        # Vectorized overlap-validity masks
        h_ok = (h[:n-1, :n-1] == 0) & (h[:n-1, 1:n] == 0) & \
               ((v[:n-1, :n-1] != v[1:n, :n-1]) | (v[:n-1, :n-1] == 0))
        v_ok = (v[:n-1, :n-1] == 0) & (v[1:n, :n-1] == 0) & \
               ((h[:n-1, :n-1] != h[:n-1, 1:n]) | (h[:n-1, :n-1] == 0))

        if all_safe:
            return np.concatenate([moves, h_ok.astype(float).flatten(), v_ok.astype(float).flatten()])

        ddown, dright = self._dangerous_edge_arrays(nf0, nf1)
        hbar_edge_safe = ~(ddown[:n-1, :n-1] | ddown[:n-1, 1:n])
        vbar_edge_safe = ~(dright[:n-1, :n-1] | dright[1:n, :n-1])

        horizontal_list = (h_ok & hbar_edge_safe).astype(float)
        vertical_list = (v_ok & vbar_edge_safe).astype(float)

        for row, col in zip(*np.where(h_ok & ~hbar_edge_safe)):
            self.horizontal_barricades[row][col] = self.barricade_id
            self.horizontal_barricades[row][col+1] = self.barricade_id
            if self.possible_path(0) and self.possible_path(1):
                horizontal_list[row][col] = 1
            self.horizontal_barricades[row][col] = 0
            self.horizontal_barricades[row][col+1] = 0

        for row, col in zip(*np.where(v_ok & ~vbar_edge_safe)):
            self.vertical_barricades[row][col] = self.barricade_id
            self.vertical_barricades[row+1][col] = self.barricade_id
            if self.possible_path(0) and self.possible_path(1):
                vertical_list[row][col] = 1
            self.vertical_barricades[row][col] = 0
            self.vertical_barricades[row+1][col] = 0

        return np.concatenate([moves, horizontal_list.flatten(), vertical_list.flatten()])


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
    

    #RL functions
    def decode_agent_action(self, action_id): #Action ID, output of DQN
        #12 directional move actions, then n-1 by n-1 hbars and n-1 by n-1 vbars.
        #Returns an array [action, actionvalue]. Action 0 = move, 1 = hbar, 2 = vbar.
        hbarvbarsize = (self.n-1)*(self.n-1)
        if action_id < N_MOVE_ACTIONS: #Move — resolve offset to absolute position
            dr, dc = MOVE_OFFSETS[action_id]
            loc = self.p1loc if self.player_turn == 0 else self.p2loc
            return [0, [loc[0]+dr, loc[1]+dc]]

        elif action_id < N_MOVE_ACTIONS + hbarvbarsize: #hbar
            action_id = action_id - N_MOVE_ACTIONS
            row = action_id // (self.n-1)
            col = action_id % (self.n-1)
            return [1, [row, col]]

        elif action_id < N_MOVE_ACTIONS + 2*hbarvbarsize: #vbar
            action_id = action_id - N_MOVE_ACTIONS - hbarvbarsize
            row = action_id // (self.n-1)
            col = action_id % (self.n-1)
            return [2, [row, col]]

        else:
            print("Invalid actionID")
            return None
        

    def agent_action_function(self, action_ID): #Function which agents will use to interact with the env.
        action = self.decode_agent_action(action_ID)
        if action[0] == 0: #Move
            self.move(action[1])
        elif action[0] == 1:
            self.place_horizontal_barrier(action[1])
        elif action[0] == 2:
            self.place_vertical_barrier(action[1])
        else:
            print("Invalid action")
            return None
        return 1
    
    def return_state_representation(self): #Returns state representation.
        # Canonical representation: always from current player's perspective.
        # Layers: hbarricades, vbarricades, my_count, opp_count, my_location, opp_location, turn
        # For player 1 the board is flipped vertically so both players always "move downward".
        p = self.player_turn
        hbarricades = torch.from_numpy((self.horizontal_barricades != 0).astype(np.float32))
        vbarricades = torch.from_numpy((self.vertical_barricades != 0).astype(np.float32))
        my_count  = torch.full((self.n, self.n), self.barricade_counts[p],   dtype=torch.float32)
        opp_count = torch.full((self.n, self.n), self.barricade_counts[1-p], dtype=torch.float32)
        my_loc  = torch.zeros((self.n, self.n), dtype=torch.float32)
        opp_loc = torch.zeros((self.n, self.n), dtype=torch.float32)
        my_pos  = self.p1loc if p == 0 else self.p2loc
        opp_pos = self.p2loc if p == 0 else self.p1loc
        my_loc[my_pos[0]][my_pos[1]]   = 1
        opp_loc[opp_pos[0]][opp_pos[1]] = 1
        turn = torch.full((self.n, self.n), p, dtype=torch.float32)

        tensor_stack = torch.stack((hbarricades, vbarricades, my_count, opp_count, my_loc, opp_loc, turn))

        # Flip vertically for player 1 so current player always moves toward row n-1
        if p == 1:
            tensor_stack = torch.flip(tensor_stack, dims=[1])

        return tensor_stack
    
    def return_action_mask(self): #Returns mask for valid actions.
        return torch.from_numpy(self.return_valid_actions_RL().astype(bool))
    
    
    def move_debug(self, player, loc): #Unsafe! For debugging

        if player == 0:
            self.p1loc = loc
        elif player == 1:
            self.p2loc = loc
        else:
            return -1
        return 0
    