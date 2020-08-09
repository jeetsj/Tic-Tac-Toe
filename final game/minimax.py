import wincheck as wc
import bestMove as bm
import numpy as np
cache = {}

# This is our Heuristic Function...
def get_legal_moves(board,team_A,team_B):
    #this will return the possible opponent moves where he has already 2 elements in a row,column or diagonal
    x_y=get_legal_moves_from_opponent(board,team_B)
    if x_y == {}:
        x_y = get_legal_moves_from_opponent(board, team_A)
        if x_y=={}:
            # this will return the possible  moves of team_A where its neigbouring elements are _'s
            il,jl=np.where(board==team_A)
            for index in range(0, len(il)):
                if jl[index] + 1 < len(board) and board[il[index], jl[index] + 1] == '_':
                    x_y[str(il[index]) + "," + str(jl[index] + 1)] = ""
                if jl[index] - 1 >= 0 and board[il[index], jl[index] - 1] == '_':
                    x_y[str(il[index]) + "," + str(jl[index] - 1)] = ""
                if il[index] - 1 >= 0 and board[il[index] - 1, jl[index]] == '_':
                    x_y[str(il[index] - 1) + "," + str(jl[index])] = ""
                if il[index] - 1 >= 0 and jl[index] - 1 >= 0 and board[il[index] - 1, jl[index] - 1] == '_':
                    x_y[str(il[index] - 1) + "," + str(jl[index] - 1)] = ""
                if il[index] - 1 >= 0 and jl[index] + 1 < len(board) and board[il[index] - 1, jl[index] + 1] == '_':
                    x_y[str(il[index] - 1) + "," + str(jl[index] + 1)] = ""
                if il[index] + 1 < len(board) and board[il[index] + 1, jl[index]] == '_':
                    x_y[str(il[index] + 1) + "," + str(jl[index])] = ""
                if il[index] + 1 < len(board) and jl[index] + 1 < len(board) and board[il[index] + 1, jl[index] + 1] == '_':
                    x_y[str(il[index] + 1) + "," + str(jl[index] + 1)] = ""
                if il[index] + 1 < len(board) and jl[index] - 1 >= 0 and board[il[index] + 1, jl[index] - 1] == '_':
                    x_y[str(il[index] + 1) + "," + str(jl[index] - 1)] = ""
    x = []
    y = []
    for k, v in x_y.items():
        x.append(int(k.split(",")[0]))
        y.append(int(k.split(",")[1]))
    legal_moves = np.array(list(zip(x, y)))
    return legal_moves

def get_legal_moves_from_opponent(board,team_B):
    il, jl = np.where(board == team_B)
    x_y = {}
    if len(il)!=0:
        for i in range(0, len(il)):
            if jl[i] + 1 < len(board) and board[il[i], jl[i] + 1] == team_B:
                if jl[i] + 2 < len(board) and board[il[i], jl[i] + 2] == "_":
                    x_y[str(il[i]) + "," + str(jl[i] + 2)] = ""
                if jl[i] - 1 >= 0 and board[il[i], jl[i] - 1] == "_":
                    x_y[str(il[i]) + "," + str(jl[i] - 1)] = ""
            if il[i] + 1 < len(board) and board[il[i] + 1, jl[i]] == team_B:
                if il[i] + 2 < len(board) and board[il[i] + 2, jl[i]] == "_":
                    x_y[str(il[i] + 2) + "," + str(jl[i])] = ""
                if il[i] - 1 >= 0 and board[il[i] - 1, jl[i]] == "_":
                    x_y[str(il[i] - 1) + "," + str(jl[i])] = ""
            if il[i] + 1 < len(board) and jl[i] + 1 < len(board) and board[il[i] + 1, jl[i] + 1] == team_B:
                if il[i] + 2 < len(board) and jl[i] + 2 < len(board) and board[il[i] + 2, jl[i] + 2] == "_":
                    x_y[str(il[i] + 2) + "," + str(jl[i] + 2)] = ""
                if il[i] - 1 >= 0 and jl[i] - 1 >= 0 and board[il[i] - 1, jl[i] - 1] == "_":
                    x_y[str(il[i] - 1) + "," + str(jl[i] - 1)] = ""
            if il[i] + 1 < len(board) and jl[i] - 1 >= 0 and board[il[i] + 1, jl[i] - 1] == team_B:
                if il[i] + 2 < len(board) and jl[i] - 2 < len(board) and board[il[i] + 2, jl[i] - 2] == "_":
                    x_y[str(il[i] + 2) + "," + str(jl[i] - 2)] = ""
                if il[i] - 1 >= 0 and jl[i] + 1 <len(board)  and board[il[i] - 1, jl[i] + 1] == "_":
                    x_y[str(il[i] - 1) + "," + str(jl[i] + 1)] = ""
    return x_y

def minimax(board, depth, ism, alpha, beta, target_len,team_A,team_B):

    score = wc.win_check(board, target_len,team_A,team_B) # Checking if Winner or Draw
    legal_moves = get_legal_moves(board,team_A,team_B)

    if score == 10:
        return score - depth

    if score == -10:
        return score + depth

    if len(legal_moves) == 0:
        return 0

    if (ism):

        # Maximising move
        best = -1000
        for i in range(0, len(legal_moves)):
            curr_move = legal_moves[i]
            board[curr_move[0], curr_move[1]] = team_A
            best = max(best, minimax(board, depth + 1, not ism, alpha, beta,target_len,team_A,team_B))
            board[curr_move[0], curr_move[1]] = '_'
            if (depth > 9):  # Depth hueristic
                return best
            alpha = max(alpha, best)
            if (alpha >= beta):
                break
        return best
    else:

        # Minimising opponent move
        best = 1000
        for i in range(0, len(legal_moves)):
            curr_move = legal_moves[i]
            board[curr_move[0], curr_move[1]] = team_B
            best = min(best, minimax(board, depth + 1, not ism, alpha, beta,target_len,team_A,team_B))
            board[curr_move[0], curr_move[1]] = '_'
            if (depth > 9):  # Depth hueristic
                return best
            beta = min(beta, best)
            if (alpha >= beta):
                break
        return best

# Only calculate a score if the board is not already in our cache.
def minimax_score_with_cache(board, depth, ism, alpha, beta, target_len,team_A,team_B):
    board_cache_key = str(board)
    if board_cache_key not in cache:
        best = minimax(board, depth, ism, alpha, beta, target_len,team_A,team_B)
        cache[board_cache_key] = best
    return cache[board_cache_key]
