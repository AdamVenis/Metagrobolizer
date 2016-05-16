## TODOS:
#   alpha beta pruning
#   hashing board states for evaluation
#   modify board instance instead of create new instance (test this)
#   use timer module to evaluate for a certain time period
##

import cmd
import copy
import random
import sys
import time

WHITE = 0
BLACK = 1
FREE = -1

class HumanAgent():
    def __init__(self, game, *args):
        self.game = game

    def move(self):
        moves = self.game.moves()
        while True:
            input = raw_input('hey buddy ').upper()
            rtn = Move.from_string(input, self.game)
            if rtn in moves:
                return rtn
            print('BAD INPUT, TRY AGAIN')

class RandomAgent():
    def __init__(self, game):
        self.game = game

    def move(self):
        moves = self.game.moves()
        return random.choice(moves)

class MinMaxAgent():
    def __init__(self, game, colour, depth):
        self.game = game
        self.colour = colour
        self.depth = depth

    def move(self):
        rtn, val = self.minmax(self.game, parity=1, depth=self.depth)
        #print('%s, %.2f' % (str(rtn), val))
        return rtn

    def minmax(self, game, parity, depth):
        m = max if parity else min
        if depth == 1:
            rtn = None
            val = -10**6 if parity else 10**6
            for move in game.moves():
                new_game = simulate(move)
                temp_val = self.static_eval(new_game) * (-1)**self.colour
                rtn, val = m((rtn, val), (move, temp_val), key=lambda x: x[1])
            return rtn, val
        else:
            rtn, val = m(((move, self.minmax(simulate(move), 1 - parity, depth - 1))
                          for move in game.moves()), key=lambda x: x[1][1])
            return rtn, val[1]

    def static_eval(self, game):
        return self.eval_player(game, WHITE) - self.eval_player(game, BLACK)

    def eval_player(self, game, colour):
        rtn = 0
        for piece in game.pieces:
            if piece.colour == colour:
                if isinstance(piece, Pawn):
                    rtn += 1
                elif isinstance(piece, Knight):
                    rtn += 3
                elif isinstance(piece, Bishop):
                    rtn += 3
                elif isinstance(piece, Rook):
                    rtn += 5
                elif isinstance(piece, Queen):
                    rtn += 9
                elif isinstance(piece, King):
                    rtn += 1000
                rtn += 0.25 * len(piece.moves(game))
        return rtn

class Game():
    def __init__(self):
        self.board = [[FREE] * 8 for _ in range(8)]
        self.pieces = []
        self.turn = 0
        self.winner = -1
        self.en_passant = None # square that can currently be en passant'd into

        for i in range(8):
            self.add(Pawn(WHITE, i, 1))
            self.add(Pawn(BLACK, i, 6))

        self.add(Rook(WHITE, 0, 0))
        self.add(Knight(WHITE, 1, 0))
        self.add(Bishop(WHITE, 2, 0))
        self.add(Queen(WHITE, 3, 0))
        self.add(King(WHITE, 4, 0))
        self.add(Bishop(WHITE, 5, 0))
        self.add(Knight(WHITE, 6, 0))
        self.add(Rook(WHITE, 7, 0))

        self.add(Rook(BLACK, 0, 7))
        self.add(Knight(BLACK, 1, 7))
        self.add(Bishop(BLACK, 2, 7))
        self.add(Queen(BLACK, 3, 7))
        self.add(King(BLACK, 4, 7))
        self.add(Bishop(BLACK, 5, 7))
        self.add(Knight(BLACK, 6, 7))
        self.add(Rook(BLACK, 7, 7))

    def add(self, piece):
        self.board[piece.x][piece.y] = piece
        self.pieces.append(piece)

    def moves(self):
        rtn = []
        for piece in self.pieces:
            if ((self.turn % 2 == 0 and piece.colour == WHITE) or
               (self.turn % 2 == 1 and piece.colour == BLACK)):
                rtn.extend(piece.moves(self))
        return rtn

    def square(self, x, y):
        if not (0 <= x < 8 and 0 <= y < 8):
            return None
        elif self.board[x][y] == FREE:
            return FREE
        else:
            return self.board[x][y].colour

    def __str__(self):
        rtn = ''
        for row in zip(*self.board)[::-1]:
            rtn += ' | '.join('.' if x == -1 else x.__repr__() for x in row) + '\n'
        return rtn


class Piece():
    def __init__(self, colour, posx, posy):
        self.colour = colour
        self.x = posx
        self.y = posy


class Pawn(Piece):
    def moves(self, game):
        rtn = []
        direction = 1 if self.colour == WHITE else -1
        if game.square(self.x, self.y + direction) == FREE:
            rtn.append(Move(game, self, self.x, self.y + direction))
            if (self.y + direction*6 in range(len(game.board)) and
                game.square(self.x, self.y + 2 * direction) == FREE):
                rtn.append(Move(game, self, self.x, self.y + 2 * direction))
            elif self.y % 8 == (-1)**direction: # second last rank
                pass # some promotion stuff
        for capture_delta in [-1, 1]:
            new_square = self.x + capture_delta, self.y + direction
            if game.square(*new_square) == 1 - self.colour or new_square == game.en_passant:
                rtn.append(Move(game, self, *new_square))

        return rtn # still needs promotions

    def __repr__(self):
        return 'P' if self.colour == WHITE else 'p'


class Knight(Piece):
    def moves(self, game):
        rtn = []
        deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for delta in deltas:
            new_x = self.x + delta[0]
            new_y = self.y + delta[1]
            if game.square(new_x, new_y) in [FREE, 1 - self.colour]:
                rtn.append(Move(game, self, new_x, new_y))
        return rtn

    def __repr__(self):
        return 'N' if self.colour == WHITE else 'n'


class Bishop(Piece):
    def moves(self, game):
        rtn = []
        deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for delta in deltas:
            new_x = self.x + delta[0]
            new_y = self.y + delta[1]
            while game.square(new_x, new_y) == FREE:
                rtn.append(Move(game, self, new_x, new_y))
                new_x += delta[0]
                new_y += delta[1]
            if game.square(new_x, new_y) == 1 - self.colour:
                rtn.append(Move(game, self, new_x, new_y))
        return rtn


    def __repr__(self):
        return 'B' if self.colour == WHITE else 'b'


class Rook(Piece):
    def moves(self, game):
        rtn = []
        deltas = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for delta in deltas:
            new_x = self.x + delta[0]
            new_y = self.y + delta[1]
            while game.square(new_x, new_y) == FREE:
                rtn.append(Move(game, self, new_x, new_y))
                new_x += delta[0]
                new_y += delta[1]
            if game.square(new_x, new_y) == 1 - self.colour:
                rtn.append(Move(game, self, new_x, new_y))
        return rtn

    def __repr__(self):
        return 'R' if self.colour == WHITE else 'r'


class Queen(Piece):
    def moves(self, game):
        rtn = []
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for delta in deltas:
            new_x = self.x + delta[0]
            new_y = self.y + delta[1]
            while game.square(new_x, new_y) == FREE:
                rtn.append(Move(game, self, new_x, new_y))
                new_x += delta[0]
                new_y += delta[1]
            if game.square(new_x, new_y) == 1 - self.colour:
                rtn.append(Move(game, self, new_x, new_y))
        return rtn

    def __repr__(self):
        return 'Q' if self.colour == WHITE else 'q'


class King(Piece):

    def __init__(self, colour, posx, posy):
        self.colour = colour
        self.x = posx
        self.y = posy
        self.can_castle = True

    def moves(self, game):
        rtn = []
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for delta in deltas:
            new_x = self.x + delta[0]
            new_y = self.y + delta[1]
            if game.square(new_x, new_y) in [FREE, 1 - self.colour]:
                rtn.append(Move(game, self, new_x, new_y))
        return rtn # cannot move into check? and through check. and castle.

    def __repr__(self):
        return 'K' if self.colour == WHITE else 'k'

class Move():
    def __init__(self, game, piece, posx, posy):
        self.game = game
        self.piece = piece
        self.posx = posx
        self.posy = posy

    def execute(self):
        if isinstance(self.game.board[self.posx][self.posy], King):
            self.game.winner = 1 - self.game.board[self.posx][self.posy].colour
        if self.game.board[self.posx][self.posy] != FREE:
            self.game.pieces.remove(self.game.board[self.posx][self.posy])

        self.game.board[self.piece.x][self.piece.y] = FREE
        self.game.board[self.posx][self.posy] = self.piece
        self.piece.x = self.posx
        self.piece.y = self.posy
        self.game.turn += 1

        if isinstance(self.piece, Pawn) and abs(self.posy - self.piece.y) == 2:
            direction = 1 if self.colour == WHITE else -1
            self.game.en_passant = (self.piece.x, self.piece.y + direction)
        else:
            self.game.en_passant = None

    @staticmethod
    def from_string(input, game):
        x_old = ord(input[0]) - 97
        y_old = int(input[1]) - 1
        x_new = ord(input[2]) - 97
        y_new = int(input[3]) - 1
        return Move(game, game.board[x_old][y_old], x_new, y_new)

    def to_string(self):
        return '%s%d%s%d' % (chr(self.piece.x + 97), self.piece.y + 1,
                             chr(self.posx + 97), self.posy + 1)

    def __eq__(self, move):
        return (move.piece == self.piece and
                move.posx == self.posx and
                move.posy == self.posy)

    def __str__(self):
        return 'Move: %s to %s%d' % (self.piece.__class__.__name__, chr(self.posx+65), self.posy+1)

def material(game, colour):
    rtn = 0
    for piece in game.pieces:
        if piece.colour == colour:
            if isinstance(piece, Pawn):
                rtn += 1
            elif isinstance(piece, Knight):
                rtn += 3
            elif isinstance(piece, Bishop):
                rtn += 3
            elif isinstance(piece, Rook):
                rtn += 5
            elif isinstance(piece, Queen):
                rtn += 9
            elif isinstance(piece, King):
                rtn += 1000
    return rtn

def simulate(move):
    new_move = copy.deepcopy(move)
    new_move.execute()
    return new_move.game

def play():
    game = Game()
    agent1 = MinMaxAgent(game, WHITE, 3)
    agent2 = MinMaxAgent(game, BLACK, 3)
    start = time.time()
    while True:
        if game.winner == 0:
            print('WHITE WINS')
            break
        elif game.winner == 1:
            print('BLACK WINS')
            break
        elif game.turn % 2 == 0:
            agent = agent1
        else:
            agent = agent2
        print(game)
        print(time.time()-start, agent.static_eval(game))
        move = agent.move()
        move.execute()

class Shell(cmd.Cmd):
    def __init__(self):
        cmd.Cmd.__init__(self, 'TAB')
        self.prompt = ''

    def do_uci(self, args):
        print('id name Metagrobolizer')
        print('id auther Adam Venis')
        print('uciok')

    def do_isready(self, args):
        print('readyok')

    def do_ucinewgame(self, args):
        pass

    def do_position(self, args):
        moves = args.split()[2:]
        game = Game()
        human_agent = HumanAgent(game)
        for move in moves:
            Move.from_string(move, game).execute()
        self.game = game

    def do_go(self, args):
        game = self.game
        engine_agent = MinMaxAgent(game, BLACK if game.turn % 2 else WHITE, 3)
        print('bestmove %s' % engine_agent.move().to_string())

    def do_quit(self, args):
        sys.exit()

if __name__ == '__main__':
    play() # terminal interface
    #Shell().cmdloop() # uci interface