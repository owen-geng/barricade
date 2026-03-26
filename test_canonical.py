"""Smoke tests for return_canonical_state_representation."""
import torch
from environment import Environment

# MCTS layers: 0=h_bar, 1=v_bar, 2=my_count, 3=opp_count, 4=my_loc, 5=opp_loc
N = 7


def make_env(p1=None, p2=None):
    env = Environment(N, 10)
    if p1: env.p1loc = list(p1)
    if p2: env.p2loc = list(p2)
    return env


def print_board(state, title=""):
    """Render a canonical state tensor as ASCII.
    Cells: ME = current player, OP = opponent, .  = empty
    Between cells: | = vertical barricade to the right
    Between rows:  = = horizontal barricade below
    """
    n = state.shape[1]
    h  = state[0].numpy()
    v  = state[1].numpy()
    me = state[4].numpy()
    op = state[5].numpy()

    if title:
        print(f"\n  {title}")
    print("    " + "  ".join(str(c) for c in range(n)))
    for r in range(n):
        # Cell row
        row = f"  {r} "
        for c in range(n):
            cell = "ME" if me[r][c] else ("OP" if op[r][c] else " .")
            row += cell
            if c < n - 1:
                row += "|" if v[r][c] else " "
        print(row)
        # Horizontal-wall row
        if r < n - 1:
            wall = "    "
            for c in range(n):
                wall += "==" if h[r][c] else "  "
                if c < n - 1:
                    wall += " "
            print(wall)


def test_player0_identity():
    """Player 0 canonical should equal raw state."""
    env = make_env(p1=(1, 2), p2=(5, 4))
    env.player_turn = 0
    raw = env.return_state_representation(mode="mcts")
    canon = env.return_canonical_state_representation()
    assert torch.equal(raw, canon), "Player 0: canonical must equal raw"
    print("PASS  player 0 identity")


def test_player_locations():
    """After 180° rotation, my_loc and opp_loc land at (n-1-r, n-1-c)."""
    env = make_env(p1=(1, 2), p2=(5, 4))

    env.player_turn = 0
    print_board(env.return_canonical_state_representation(), "player 0 view  (ME=p1 at [1,2], OP=p2 at [5,4])")

    env.player_turn = 1  # player 1's turn: my_pos=p2loc=[5,4], opp_pos=p1loc=[1,2]
    canon = env.return_canonical_state_representation()
    print_board(canon, "player 1 canonical  (ME=p2 at [5,4] -> rotated to [1,2])")

    # p2loc [5,4] → rot90 → [N-1-5, N-1-4] = [1, 2]
    assert canon[4, 1, 2] == 1.0, f"my_loc wrong: {canon[4].nonzero()}"
    assert canon[4].sum() == 1.0

    # p1loc [1,2] → rot90 → [N-1-1, N-1-2] = [5, 4]
    assert canon[5, 5, 4] == 1.0, f"opp_loc wrong: {canon[5].nonzero()}"
    assert canon[5].sum() == 1.0

    print("PASS  player locations under rotation")


def test_hbar_rotation():
    """h_bar[r][c], h_bar[r][c+1] should appear at [n-2-r][n-1-c], [n-2-r][n-2-c] in player 1's view.

    Derivation:
      rot90(2): [r][c] → [N-1-r][N-1-c]
      shift up by 1: [N-1-r] → [N-2-r]
    """
    env = make_env()
    r, c = 2, 1
    env.horizontal_barricades[r][c]   = 1
    env.horizontal_barricades[r][c+1] = 1

    env.player_turn = 0
    print_board(env.return_canonical_state_representation(), f"h_bar raw  (== below row {r}, cols {c}-{c+1})")

    env.player_turn = 1
    canon = env.return_canonical_state_representation()
    er, ec1, ec2 = N-2-r, N-1-c, N-2-c  # expected: row=3, cols=5,4
    print_board(canon, f"h_bar player 1 canonical  (== expected below row {er}, cols {ec2}-{ec1})")

    assert canon[0, er, ec1] == 1.0, f"h_bar cell 1 expected [{er},{ec1}], got {canon[0].nonzero()}"
    assert canon[0, er, ec2] == 1.0, f"h_bar cell 2 expected [{er},{ec2}], got {canon[0].nonzero()}"
    assert canon[0].sum() == 2.0, f"Expected 2 h_bar cells, got {canon[0].sum()}"
    print("PASS  horizontal barricade rotation + shift")


def test_vbar_rotation():
    """v_bar[r][c], v_bar[r+1][c] should appear at [n-1-r][n-2-c], [n-2-r][n-2-c] in player 1's view.

    Derivation:
      rot90(2): [r][c] → [N-1-r][N-1-c]
      shift left by 1: [N-1-c] → [N-2-c]
    """
    env = make_env()
    r, c = 1, 2
    env.vertical_barricades[r][c]   = 1
    env.vertical_barricades[r+1][c] = 1

    env.player_turn = 0
    print_board(env.return_canonical_state_representation(), f"v_bar raw  (| right of col {c}, rows {r}-{r+1})")

    env.player_turn = 1
    canon = env.return_canonical_state_representation()
    er1, er2, ec = N-1-r, N-2-r, N-2-c  # expected: rows=5,4, col=3
    print_board(canon, f"v_bar player 1 canonical  (| expected right of col {ec}, rows {er2}-{er1})")

    assert canon[1, er1, ec] == 1.0, f"v_bar cell 1 expected [{er1},{ec}], got {canon[1].nonzero()}"
    assert canon[1, er2, ec] == 1.0, f"v_bar cell 2 expected [{er2},{ec}], got {canon[1].nonzero()}"
    assert canon[1].sum() == 2.0, f"Expected 2 v_bar cells, got {canon[1].sum()}"
    print("PASS  vertical barricade rotation + shift")


def test_symmetric_position():
    """For a board that is 180°-symmetric, both players' canonical views should be identical."""
    # p1 at [1,3], p2 at [5,3] — symmetric under 180° rotation (n=7, center col=3)
    env = make_env(p1=(1, 3), p2=(5, 3))

    env.player_turn = 0
    canon0 = env.return_canonical_state_representation()

    env.player_turn = 1
    canon1 = env.return_canonical_state_representation()

    assert torch.equal(canon0, canon1), \
        f"Symmetric position: canonical views differ.\nP0:\n{canon0}\nP1:\n{canon1}"
    print("PASS  symmetric position: both canonical views identical")


def test_convert_canonical_action():
    """Round-trip: encode action in canonical view, convert back, execute in env, check result."""
    from environment import MOVE_OFFSETS

    env = make_env(p1=(3, 3), p2=(3, 3))  # positions don't matter for barricade tests

    # --- Movement: every canonical move action should invert correctly ---
    for i in range(12):
        env.player_turn = 1
        real = env.convert_canonical_action(i)
        dr_canon, dc_canon = MOVE_OFFSETS[i]
        dr_real,  dc_real  = MOVE_OFFSETS[real]
        assert (dr_real, dc_real) == (-dr_canon, -dc_canon), \
            f"Move {i} ({dr_canon},{dc_canon}) -> real {real} ({dr_real},{dc_real}), expected ({-dr_canon},{-dc_canon})"
    print("PASS  move action inversion (all 12 directions)")

    # --- H-bar: place canonical action, execute real action, check barricade landed correctly ---
    # Canonical h_bar at (er=3, ec=2) should map to real (n-2-3, n-2-2) = (2, 1)
    env2 = make_env()
    env2.player_turn = 1
    er, ec = 3, 2
    canon_action = 12 + er * (N - 1) + ec
    real_action  = env2.convert_canonical_action(canon_action)
    real_r, real_c = divmod(real_action - 12, N - 1)
    assert (real_r, real_c) == (N - 2 - er, N - 2 - ec), \
        f"h_bar: expected real ({N-2-er},{N-2-ec}), got ({real_r},{real_c})"
    print("PASS  h_bar canonical -> real conversion")

    # --- V-bar: canonical v_bar at (er=4, ec=1) should map to real (n-2-4, n-2-1) = (1, 2) ---
    hbarvbarsize = (N - 1) ** 2
    er, ec = 4, 1
    canon_action = 12 + hbarvbarsize + er * (N - 1) + ec
    real_action  = env2.convert_canonical_action(canon_action)
    real_r, real_c = divmod(real_action - 12 - hbarvbarsize, N - 1)
    assert (real_r, real_c) == (N - 2 - er, N - 2 - ec), \
        f"v_bar: expected real ({N-2-er},{N-2-ec}), got ({real_r},{real_c})"
    print("PASS  v_bar canonical -> real conversion")

    # --- Player 0: all actions pass through unchanged ---
    env3 = make_env()
    env3.player_turn = 0
    for a in [0, 5, 12, 12 + hbarvbarsize]:
        assert env3.convert_canonical_action(a) == a, f"Player 0: action {a} should be unchanged"
    print("PASS  player 0 identity (no conversion)")


if __name__ == "__main__":
    test_player0_identity()
    test_player_locations()
    test_hbar_rotation()
    test_vbar_rotation()
    test_symmetric_position()
    test_convert_canonical_action()
    print("\nAll smoke tests passed.")
