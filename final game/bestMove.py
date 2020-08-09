import minimax as mm

class Game:
    row = 0
    col = 0

# this function will do the min max algo and return the best move
def find_best_move(board, alpha, beta, target_len,team_A,team_B):

    bestVal = -1000
    bestMove = Game()
    bestMove.row = -1
    bestMove.col = -1
    legal_moves = mm.get_legal_moves(board,team_A,team_B)
    print(len(legal_moves))
    print(legal_moves)
    for i in range(0, len(legal_moves)):
        curr_move = legal_moves[i]
        board[curr_move[0], curr_move[1]] = team_A
        moveVal = mm.minimax_score_with_cache(board, 0, False, alpha, beta, target_len,team_A,team_B)
        board[curr_move[0], curr_move[1]] = '_'
        if moveVal > bestVal:
            bestMove.row = curr_move[0]
            bestMove.col = curr_move[1]
            bestVal = moveVal
    return bestMove.row, bestMove.col
