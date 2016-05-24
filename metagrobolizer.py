## TODOS:
#   alpha beta pruning
#   remove recursion
#   modify board instance instead of create new instance (test this)
#   use timer module to evaluate for a certain time period
#   make the agent stop playing with its food (most direct checkmate)
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
    def __init__(self, *args):
        pass

    def move(self, game):
        moves = game.moves()
        while True:
            input = raw_input('hey buddy tell me your move: ').lower()
            rtn = Move.from_string(input, game)
            if rtn in moves:
                return rtn
            print('BAD INPUT, TRY AGAIN')

class RandomAgent():
    def move(self, game):
        return random.choice(game.moves())

class MinMaxAgent():
    def __init__(self, depth):
        self.depth = depth

    def move(self, game):
        results = [(-10**6 * (-1)**i, None) for i in range(self.depth + 1)]
        nodes = [[(simulate(move), [move]) for move in game.moves()]]
        eval_cache = {}
        ehits = 0
        while nodes:
            # expand(nodes)
            while len(nodes) < self.depth:
                while not nodes[-1]:
                    del nodes[-1]
                g, m = nodes[-1].pop()
                nodes.append([(simulate(move), m + [move]) for move in g.moves()])
            flag = True
            while flag:
                # evaluate deepest node
                g, m = nodes[-1].pop()
                fen = g.to_FEN()
                if fen not in eval_cache:
                    eval_cache[fen] = (self.static_eval(g)*(-1)**game.turn, m)
                else:
                    ehits += 1
                results[-1] = eval_cache[fen]

                # update running results
                i = self.depth - 1
                while i >= 0:
                    minmax = max if i % 2 == 0 else min
                    results[i] = minmax(results[i], results[i + 1])
                    if i > 0 and results[i] == minmax(results[i], results[i-1]):
                        for j in range(i, len(nodes)):
                            nodes[j] = []
                    results[i + 1] = (10**6 * (-1)**i, None)
                    if nodes[i]:
                        break   
                    flag = False
                    del nodes[i]
                    i -= 1
        print('eval cache size: %s with %d hits' % (len(eval_cache), ehits))
        print('expected sequence: %s' % [str(i) for i in results[0][1]])
        return results[0][1][0]

    def static_eval(self, game):
        if game.winner == WHITE:
            return 1000
        elif game.winner == BLACK:
            return -1000
        else:
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
                    rtn += 0.1 * len(Game.piece_objs[piece](game, x, y, c))
        return rtn

def pawn(game, x, y, colour):
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


def knight(game, x, y, colour):
    rtn = []
    deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    for delta in deltas:
        new_x = x + delta[0]
        new_y = y + delta[1]
        if game.square(new_x, new_y) in [FREE, 1 - colour]:
            rtn.append(Move(game, x, y, new_x, new_y))
    return rtn

def bishop(game, x, y, colour):
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

def rook(game, x, y, colour):
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

def queen(game, x, y, colour):
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

def king(game, x, y, colour):
    rtn = []
    deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for delta in deltas:
        new_x = x + delta[0]
        new_y = y + delta[1]
        if game.square(new_x, new_y) in [FREE, 1 - colour]:
            rtn.append(Move(game, x, y, new_x, new_y))
    return rtn # cannot move into check? and through check. and castle.

class Game():
    piece_objs = {}
    piece_objs['p'] = piece_objs['P'] = pawn
    piece_objs['n'] = piece_objs['N'] = knight
    piece_objs['b'] = piece_objs['B'] = bishop
    piece_objs['r'] = piece_objs['R'] = rook
    piece_objs['q'] = piece_objs['Q'] = queen
    piece_objs['k'] = piece_objs['K'] = king

    def __init__(self, other=None):
        if other:
            self.board = [[piece for piece in row] for row in other.board]
            self.pieces = {k: [p for p in v] for k, v in other.pieces.iteritems()}
            self.turn = other.turn
            self.winner = other.winner
            self.castling = other.castling
            self.castling_trail = other.castling_trail
            self.en_passant = other.en_passant
            self.halfmove_clock = other.halfmove_clock
        else:
            self.board = [[FREE] * 8 for _ in range(8)]
            self.pieces = {}
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
        self.pieces.setdefault(piece, []).append((x, y, ord(piece) > 97))

    def moves(self):
        rtn = []
        for piece in self.pieces:
            for x, y, c in self.pieces[piece]:
                if c == self.turn % 2:
                    rtn.extend(Game.piece_objs[piece](self, x, y, c))
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
                                      ('%s%d' % (chr(self.en_passant[0] + 97), self.en_passant[1] + 1) if
                                        self.en_passant else '-'),
                                      self.halfmove_clock,
                                      self.turn // 2 + 1)

    def __str__(self):
        rtn = ''
        for row in zip(*self.board)[::-1]:
            rtn += ' | '.join('.' if x == -1 else x for x in row) + '\n'
        return rtn

    def __eq__(self, other):
        return str(self) == str(other)

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

        if self.source_piece.lower() == 'p' and abs(self.new_y - self.old_y) == 2:
            self.game.en_passant = (self.old_x, (self.old_y + self.new_y) // 2)
        else:
            self.game.en_passant = None

        self.game.turn += 1

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

    def to_SAN(self): # incomplete "short algebraic notation"
        piece_letter = self.game.board[self.old_x][self.old_y].upper()
        rtn = ''
        if piece_letter != 'P':
            rtn += piece_letter
        if self.game.board[self.new_x][self.new_y] != FREE:
            if piece_letter == 'P':
                rtn += chr(self.old_x + 97)
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
    new_move = copy.copy(move)
    new_move.game = Game(move.game)
    new_move.execute()
    return new_move.game

def play(pgn_output_file=None):
    game = Game()
    agent1 = MinMaxAgent(4)
    agent2 = MinMaxAgent(4)
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
        print('Time: %s, Eval: %.2f' % (time.time()-start, agent.static_eval(game)))
        move = agent.move(game)

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
        self.game = Game()
        human_agent = HumanAgent()
        for move in moves:
            Move.from_string(move, self.game).execute()

    def do_go(self, args):
        engine_agent = MinMaxAgent(3)
        print('bestmove %s' % engine_agent.move(self.game).to_string())

    def do_quit(self, args):
        sys.exit()

if __name__ == '__main__':
    #play()
    play(pgn_output_file='replays/replay004.pgn') # terminal interface
    #Shell().cmdloop() # uci interface
