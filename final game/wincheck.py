import bestMove as bm
import numpy as np

#this will return 10 if X or 0's successfully formed sequence with target_len or else it will return -10
def win_check(board, target_len,team_A,team_B):
    a = team_A * target_len
    b = team_B * target_len

    # Checking for Rows for X or O victory
    for row in range(0, len(board)):
        full_row = "".join(board[row])
        if a in full_row:
            return +10
        elif b in full_row:
            return -10

    # Checking for Col for X or O victory
    for col in range(0, len(board)):
        full_col = "".join(board[:, col])
        if a in full_col:
            return +10
        elif b in full_col:
            return -10

    # Checking for Diagonals for X or O victory
    diag = "".join(board.diagonal())
    rev_diag = "".join(np.diag(np.fliplr(board)))
    if a in diag or a in rev_diag:
        return 10
    elif a in diag or a in rev_diag:
        return -10

    # checking all other possible diagonals for sub matrices
    l = len(board) - 3
    for i in range(4, len(board) + 1):
        matrix1_diagonal = "".join(board[:i, l:].diagonal())
        matrix2_diagonal = "".join(board[l:, :i - 1].diagonal())

        # 3 and 4 we need opposite diagonal
        matrix3_diagonal = "".join(np.diag(np.fliplr(board[:i - 1, :i - 1])))
        matrix4_diagonal = "".join(np.diag(np.fliplr(board[l:, l:])))
        if a in matrix1_diagonal or a in matrix2_diagonal or a in matrix3_diagonal or a in matrix4_diagonal:
            return 10
        elif b in matrix1_diagonal or b in matrix2_diagonal or b in matrix3_diagonal or b in matrix4_diagonal:
            return -10
        l -= 1
