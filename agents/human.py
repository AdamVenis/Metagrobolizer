from game import *

class HumanAgent():
    def __init__(self, *args):
        pass

    def move(self, game):
        moves = game.moves()
        while True:
            user_input = input('hey buddy tell me your move: ').lower()
            rtn = Move.from_string(user_input, game)
            if rtn in moves:
                return rtn
            print('BAD INPUT, TRY AGAIN')