import bestMove as bm
import time
import numpy as np
import sys
import API.GetBoardAPI as gb
import API.GetMoveAPI as gm
import API.MakeAMoveAPI as mmv
import json

# This will return the current board details.
def get_board(move,gameId):
    data = gb.get_current_board(move, gameId)
    data = json.loads(data)
    rows = cols = 12
    board = np.full((rows, cols), "_")
    if data["output"]:
        board_data=json.loads(data["output"])
        # rows=cols=data["boardSize"]
        for k, v in board_data.items():
            board[int(k.split(",")[0]), int(k.split(",")[1])] = v
    target=data["target"]

    return board,target

# This is used to check if opponent has placed his move or not
def check_move(gameId):
    data = gm.get_prev_move(gameId)
    data = json.loads(data)
    return data["moves"][0]["teamId"]

# This is the main method we call and returns i,j index for the matrix
if __name__ == '__main__':

    gameId="928"
    opponent_teamid = "1213"
    team_A = 'O' #My Team
    team_B = 'X' #Opponent Team
    board, target = get_board("0,4", gameId)

    alpha = -sys.maxsize
    beta = sys.maxsize
    while(True):

        if np.all(board == board[0, :]):
            mmv.make_a_move(str(6) + "," + str(6), gameId)
            board, target = get_board("0,4", gameId)
        else:
            last_move = check_move(gameId)
            if last_move == opponent_teamid :
                startTime = time.time()
                board, target = get_board("0,4", gameId)
                row, col = bm.find_best_move(board, alpha, beta, target,team_A,team_B)
                end = time.time()
                print(end - startTime)
                print(row, col)
                data = mmv.make_a_move(str(row) + "," + str(col), gameId)
                board, target = get_board("0,4", gameId)
                print(board)
                data = json.loads(data)
                # Loop Termination Condition
                if "message" in data.keys() and data["message"] == "Cannot make move - Game is no longer open: " + gameId + "":
                    break

            time.sleep(3)