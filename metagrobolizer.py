## TODOS:
#   alpha beta pruning
#   remove recursion
#   modify board instance instead of create new instance (test this)
#   use timer module to evaluate for a certain time period
#   remove state from piece objects so they don't need to be instantiated
##

import cmd
import copy
import random
import sys
import time

from collections import defaultdict

FREE = -1
WHITE = 0
BLACK = 1
PAWN = 2
KNIGHT = 3
BISHOP = 4
ROOK = 5
QUEEN = 6
KING = 7

class HumanAgent():
    def __init__(self, game, *args):
        self.game = game

    def move(self):
        moves = self.game.moves()
        while True:
            input = raw_input('hey buddy ').lower()
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
        self.cache = {}
        self.instances = defaultdict(int)
        rtn, val = self.minmax(self.game, parity=1, depth=self.depth)
        print('instances: %s, cache_size: %s' % (self.instances, len(self.cache)))
        #print('%s, %.2f' % (str(rtn), val))
        return rtn

    def minmax(self, game, parity, depth):
        self.instances[depth] += 1
        if game.to_FEN() in self.cache:
            return self.cache[game.to_FEN()]

        m = max if parity else min
        if depth == 1:
            rtn = None
            val = -10**6 if parity else 10**6
            for move in game.moves():
                new_game = simulate(move)
                self.instances[0] += 1
                if new_game.to_FEN() in self.cache:
                    temp_val = self.cache[new_game.to_FEN()]
                else:
                    temp_val = self.static_eval(new_game) * (-1)**self.colour
                    self.cache[new_game.to_FEN()] = temp_val
                rtn, val = m((rtn, val), (move, temp_val), key=lambda x: x[1])
            self.cache[game.to_FEN()] = val
            return rtn, val
        else:
            rtn, val = m(((move, self.minmax(simulate(move), 1 - parity, depth - 1))
                          for move in game.moves()), key=lambda x: x[1][1])
            self.cache[game.to_FEN()] = val
            return rtn, val[1]

    def static_eval(self, game):
        return self.eval_player(game, WHITE) - self.eval_player(game, BLACK)

    def eval_player(self, game, colour):
        rtn = 0
        for piece in game.pieces:
            if colour == (ord(piece) > 97):
                if piece in ['p', 'P']:
                    rtn += 1 * len(game.pieces[piece])
                elif piece in ['n', 'N']:
                    rtn += 3 * len(game.pieces[piece])
                elif piece in ['b', 'B']:
                    rtn += 3 * len(game.pieces[piece])
                elif piece in ['r', 'R']:
                    rtn += 5 * len(game.pieces[piece])
                elif piece in ['q', 'Q']:
                    rtn += 9 * len(game.pieces[piece])
                elif piece in ['k', 'K']:
                    rtn += 1000 * len(game.pieces[piece])
                for x, y, c in game.pieces[piece]:
                    rtn += 0.25 * len(game.piece_objs(piece).moves(game, x, y, c))
        return rtn

class Game():
    def __init__(self):
        self.board = [[FREE] * 8 for _ in range(8)]
        self.pieces = defaultdict(list)
        self.turn = 0
        self.winner = -1
        self.castling = 'KQkq' # FEN notation for each castle possibility
        self.castling_trail = [] # for avoiding castling through check
        self.en_passant = None # square that can currently be en passant'd into
        self.halfmove_clock = 0

        for i in range(8):
            self.add('P', i, 1)
            self.add('p', i, 6)

        self.add('R', 0, 0)
        self.add('N', 1, 0)
        self.add('B', 2, 0)
        self.add('Q', 3, 0)
        self.add('K', 4, 0)
        self.add('B', 5, 0)
        self.add('N', 6, 0)
        self.add('R', 7, 0)

        self.add('r', 0, 7)
        self.add('n', 1, 7)
        self.add('b', 2, 7)
        self.add('q', 3, 7)
        self.add('k', 4, 7)
        self.add('b', 5, 7)
        self.add('n', 6, 7)
        self.add('r', 7, 7)

    def add(self, piece, x, y):
        self.board[x][y] = piece
        self.pieces[piece].append((x, y, ord(piece) > 97))

    def piece_objs(self, piece):
        piece = piece.lower()
        if piece == 'p':
            return Pawn
        elif piece == 'n':
            return Knight
        elif piece == 'b':
            return Bishop
        elif piece == 'r':
            return Rook
        elif piece == 'q':
            return Queen
        elif piece == 'k':
            return King

    def moves(self):
        rtn = []
        for piece in self.pieces:
            for x, y, c in self.pieces[piece]:
                if c == self.turn % 2:
                    rtn.extend(self.piece_objs(piece).moves(self, x, y, c))
        return rtn

    def square(self, x, y):
        if not (0 <= x < 8 and 0 <= y < 8):
            return None
        elif self.board[x][y] == FREE:
            return FREE
        else:
            return ord(self.board[x][y]) > 97

    def to_FEN(self):
        FEN_board = []
        space_counter = 0
        for row in zip(*self.board)[::-1]:
            FEN_board.append('')
            for col in row:
                if col == FREE:
                    space_counter += 1
                else:
                    if space_counter > 0:
                        FEN_board[-1] += str(space_counter)
                        space_counter = 0
                    FEN_board[-1] += col
            if space_counter > 0:
                FEN_board[-1] += str(space_counter)
                space_counter = 0
        FEN_board = '/'.join(FEN_board)
        return '%s %s %s %s %d %d' % (FEN_board,
                                      'b' if self.turn % 2 else 'w',
                                      self.castling,
                                      self.en_passant or '-',
                                      self.halfmove_clock,
                                      self.turn // 2 + 1)

    def __str__(self):
        rtn = ''
        for row in zip(*self.board)[::-1]:
            rtn += ' | '.join('.' if x == -1 else x for x in row) + '\n'
        return rtn

    def __eq__(self, other):
        return str(self) == str(other)


class Piece():
    def __init__(self, colour, posx, posy):
        self.colour = colour
        self.x = posx
        self.y = posy


class Pawn(Piece):
    @staticmethod
    def moves(game, x, y, colour):
        rtn = []
        direction = 1 if colour == WHITE else -1
        if game.square(x, y + direction) == FREE:
            rtn.append(Move(game, x, y, x, y + direction))
            if (y + direction*6 in range(len(game.board)) and
                game.square(x, y + 2 * direction) == FREE):
                rtn.append(Move(game, x, y, x, y + 2 * direction))
            elif y % 8 == (-1)**direction: # second last rank
                pass # some promotion stuff
        for capture_delta in [-1, 1]:
            new_square = x + capture_delta, y + direction
            if game.square(*new_square) == 1 - colour or new_square == game.en_passant:
                rtn.append(Move(game, x, y, *new_square))

        return rtn # still needs promotions

    def __repr__(self):
        return 'P' if self.colour == WHITE else 'p'


class Knight(Piece):
    @staticmethod
    def moves(game, x, y, colour):
        rtn = []
        deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for delta in deltas:
            new_x = x + delta[0]
            new_y = y + delta[1]
            if game.square(new_x, new_y) in [FREE, 1 - colour]:
                rtn.append(Move(game, x, y, new_x, new_y))
        return rtn

    def __repr__(self):
        return 'N' if self.colour == WHITE else 'n'


class Bishop(Piece):
    @staticmethod
    def moves(game, x, y, colour):
        rtn = []
        deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for delta in deltas:
            new_x = x + delta[0]
            new_y = y + delta[1]
            while game.square(new_x, new_y) == FREE:
                rtn.append(Move(game, x, y, new_x, new_y))
                new_x += delta[0]
                new_y += delta[1]
            if game.square(new_x, new_y) == 1 - colour:
                rtn.append(Move(game, x, y, new_x, new_y))
        return rtn


    def __repr__(self):
        return 'B' if self.colour == WHITE else 'b'


class Rook(Piece):
    @staticmethod
    def moves(game, x, y, colour):
        rtn = []
        deltas = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        for delta in deltas:
            new_x = x + delta[0]
            new_y = y + delta[1]
            while game.square(new_x, new_y) == FREE:
                rtn.append(Move(game, x, y, new_x, new_y))
                new_x += delta[0]
                new_y += delta[1]
            if game.square(new_x, new_y) == 1 - colour:
                rtn.append(Move(game, x, y, new_x, new_y))
        return rtn

    def __repr__(self):
        return 'R' if self.colour == WHITE else 'r'


class Queen(Piece):
    @staticmethod
    def moves(game, x, y, colour):
        rtn = []
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for delta in deltas:
            new_x = x + delta[0]
            new_y = y + delta[1]
            while game.square(new_x, new_y) == FREE:
                rtn.append(Move(game, x, y, new_x, new_y))
                new_x += delta[0]
                new_y += delta[1]
            if game.square(new_x, new_y) == 1 - colour:
                rtn.append(Move(game, x, y, new_x, new_y))
        return rtn

    def __repr__(self):
        return 'Q' if self.colour == WHITE else 'q'


class King(Piece):
    @staticmethod
    def moves(game, x, y, colour):
        rtn = []
        deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for delta in deltas:
            new_x = x + delta[0]
            new_y = y + delta[1]
            if game.square(new_x, new_y) in [FREE, 1 - colour]:
                rtn.append(Move(game, x, y, new_x, new_y))
        return rtn # cannot move into check? and through check. and castle.

    def __repr__(self):
        return 'K' if self.colour == WHITE else 'k'

class Move():
    def __init__(self, game, old_x, old_y, new_x, new_y):
        self.game = game
        self.old_x = old_x
        self.old_y = old_y
        self.new_x = new_x
        self.new_y = new_y
        self.source_piece = game.board[old_x][old_y]
        self.target_piece = game.board[new_x][new_y]

    def execute(self):
        if self.target_piece in ['k', 'K']:
            self.game.winner = 1 - (ord(self.target_piece) > 97)
        if self.target_piece != FREE:
            self.game.pieces[self.target_piece].remove(
                (self.new_x, self.new_y, ord(self.target_piece) > 97))

        self.game.board[self.old_x][self.old_y] = FREE
        self.game.board[self.new_x][self.new_y] = self.source_piece
        self.game.pieces[self.source_piece].remove((self.old_x, self.old_y, ord(self.source_piece) > 97))
        self.game.pieces[self.source_piece].append((self.new_x, self.new_y, ord(self.source_piece) > 97))

        self.game.turn += 1

        if self.source_piece.lower() == 'p' and abs(self.new_y - self.old_y) == 2:
            direction = 1 if self.game.turn % 2 == 0 else -1
            self.game.en_passant = (self.old_x, self.old_y + direction)
        else:
            self.game.en_passant = None

    @staticmethod
    def from_string(input, game):
        old_x = ord(input[0]) - 97
        old_y = int(input[1]) - 1
        new_x = ord(input[2]) - 97
        new_y = int(input[3]) - 1
        return Move(game, old_x, old_y, new_x, new_y)

    def to_string(self):
        return '%s%d%s%d' % (chr(self.old_x + 97), self.old_y + 1,
                             chr(self.new_x + 97), self.new_y + 1)

    def to_SAN(self): # incomplete
        piece_letter = repr(self.piece).upper()
        rtn = '' if piece_letter == 'P' else piece_letter
        if self.game.board[self.new_x][self.new_y] != FREE:
            rtn += 'x'
        rtn += self.to_string()[2:]
        return rtn

    def __eq__(self, move):
        return (move.old_x == self.old_x and
                move.old_y == self.old_y and
                move.new_x == self.new_x and
                move.new_y == self.new_y)

    def __str__(self):
        return 'Move: %s to %s%d' % (self.source_piece, chr(self.new_x + 65), self.new_y + 1)

def simulate(move):
    new_move = copy.deepcopy(move)
    new_move.execute()
    return new_move.game

def play(pgn_output_file=None):
    game = Game()
    agent1 = MinMaxAgent(game, WHITE, 3)
    agent2 = MinMaxAgent(game, BLACK, 3)
    start = time.time()
    if pgn_output_file:
        out = open(pgn_output_file, 'w+')
        out.write(
'''[Event "Simulation"]
[Site "Cyberspace"]
[Date "2016.05.16"]
[White "Metagrobolizer"]
[Black "Metagrobolizer"]
[Result "1/2-1/2"]

''')
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
        print(game.to_FEN())
        print('Time: %s, Eval: %.2f' % (time.time()-start, agent.static_eval(game)))
        move = agent.move()

        if pgn_output_file:
            if game.turn % 2 == 0:
                out.write('%d. ' % (game.turn // 2 + 1))
            out.write('%s ' % move.to_SAN())

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
    play()
    #play(pgn_output_file='replays/replay002.pgn') # terminal interface
    #Shell().cmdloop() # uci interface