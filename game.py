
import copy

from operator import itemgetter

FREE = -1
WHITE = 0
BLACK = 1
PAWNS = ['p', 'P']
KNIGHTS = ['n', 'N']
BISHOPS = ['b', 'B']
ROOKS = ['r', 'R']
QUEENS = ['q', 'Q']
KINGS = ['k', 'K']
CORNERS = [(0, 0), (0, 7), (7, 0), (7, 7)]
PIECE_VALS = {'p': 1, 'P': 1, 'n': 3, 'N': 3,
              'b': 3, 'B': 3, 'r': 5, 'R': 5,
              'q': 9, 'Q': 9, 'k': 1000, 'K': 1000}

def pawn(game, x, y, colour):
    rtn = []
    direction = 1 if colour == WHITE else -1
    if game.board[x][y + direction] == FREE:
        if (y + direction) % 7 == 0: # second last rank
            for i in ['NBRQ', 'nbrq'][colour]:
                rtn.append(Move(game, x, y, x, y + direction, i))
        else:
            rtn.append(Move(game, x, y, x, y + direction))
            if (((y - direction) % 7 == 0) and
                game.board[x][y + 2 * direction] == FREE):
                rtn.append(Move(game, x, y, x, y + 2 * direction))
    for capture_delta in [-1, 1]:
        new_square = x + capture_delta, y + direction
        if new_square == game.en_passant:
            rtn.append(Move(game, x, y, *new_square))
        elif new_square in game.castling_trail or game.square(*new_square) == 1 - colour:
            if (y + direction) % 7 == 0:
                for i in ['NBRQ', 'nbrq'][colour]:
                    rtn.append(Move(game, x, y, x + capture_delta, y + direction, i))
            else:
                rtn.append(Move(game, x, y, *new_square))
    return rtn


def knight(game, x, y, colour):
    rtn = []
    deltas = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
    for dx, dy in deltas:
        new_x = x + dx
        new_y = y + dy
        if game.square(new_x, new_y) in [FREE, 1 - colour]:
            rtn.append(Move(game, x, y, new_x, new_y))
    return rtn

def bishop(game, x, y, colour):
    rtn = []
    deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for dx, dy in deltas:
        new_x = x + dx
        new_y = y + dy
        while game.square(new_x, new_y) == FREE:
            rtn.append(Move(game, x, y, new_x, new_y))
            new_x += dx
            new_y += dy
        if game.square(new_x, new_y) == 1 - colour:
            rtn.append(Move(game, x, y, new_x, new_y))
    return rtn

def rook(game, x, y, colour):
    rtn = []
    deltas = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    for dx, dy in deltas:
        new_x = x + dx
        new_y = y + dy
        while game.square(new_x, new_y) == FREE:
            rtn.append(Move(game, x, y, new_x, new_y))
            new_x += dx
            new_y += dy
        if game.square(new_x, new_y) == 1 - colour:
            rtn.append(Move(game, x, y, new_x, new_y))
    return rtn

def queen(game, x, y, colour):
    rtn = []
    deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dx, dy in deltas:
        new_x = x + dx
        new_y = y + dy
        while game.square(new_x, new_y) == FREE:
            rtn.append(Move(game, x, y, new_x, new_y))
            new_x += dx
            new_y += dy
        if game.square(new_x, new_y) == 1 - colour:
            rtn.append(Move(game, x, y, new_x, new_y))
    return rtn

def king(game, x, y, colour):
    rtn = []
    deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dx, dy in deltas:
        new_x = x + dx
        new_y = y + dy
        if game.square(new_x, new_y) in [FREE, 1 - colour]:
            rtn.append(Move(game, x, y, new_x, new_y))
    if (chr(75 + 32 * colour) in game.castling and
        game.square(x + 1, y) == FREE and
        game.square(x + 2, y) == FREE):
        rtn.append(Move(game, x, y, x + 2, y))
    if (chr(81 + 32 * colour) in game.castling and
        game.square(x - 1, y) == FREE and
        game.square(x - 2, y) == FREE and
        game.square(x - 3, y) == FREE):
        rtn.append(Move(game, x, y, x - 2, y))
    return rtn # cannot move into check? and through check. and castle.

PIECE_FNS = {'p': pawn, 'P': pawn, 'n': knight, 'N': knight,
              'b': bishop, 'B': bishop, 'r': rook, 'R': rook,
              'q': queen, 'Q': queen, 'k': king, 'K': king}

class Game():
    def __init__(self, other=None):
        if other:
            self.board = [[piece for piece in row] for row in other.board]
            self.pieces = {k: list(v) for k, v in other.pieces.items()}
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
                self.add('RNBQKBNR'[i], i, 0)
                self.add('rnbqkbnr'[i], i, 7)

    def add(self, piece, x, y):
        self.board[x][y] = piece
        self.pieces.setdefault(piece, []).append((x, y, ord(piece) > 97))

    def moves(self):
        if self.winner != -1:
            return []
        rtn = []
        for piece in self.pieces:
            for x, y, c in self.pieces[piece]:
                if c == self.turn % 2:
                    rtn.extend(PIECE_FNS[piece](self, x, y, c))
        return rtn

    def square(self, x, y):
        if x < 0 or x >= 8 or y < 0 or y >= 8:
            return None
        elif self.board[x][y] == FREE:
            return FREE
        else:
            return ord(self.board[x][y]) > 97

    def key(self):
        return ''.join(str(row) for row in self.board)

    def to_FEN(self):
        FEN_board = []
        space_counter = 0
        for row in zip(*reversed(self.board)):
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
                                      sq2str(*self.en_passant) if self.en_passant else '-',
                                      self.halfmove_clock,
                                      self.turn // 2 + 1)

    @staticmethod
    def from_SAN(SAN_str):
        # utility method for debugging (can be used to load a game from a PGN file)
        g = Game()
        turns = SAN_str.split('.')[1:]
        try:
            for move in [m for turn in turns for m in turn.split()[:2]]:
                print(str(move))
                for m in g.moves():
                    new_square = sq2str(m.new_x, m.new_y)
                    if (move == 'O-O' and g.board[m.old_x][m.old_y] in KINGS
                        and m.old_x + 2 == m.new_x):
                        m.execute()
                    elif (move == 'O-O-O' and g.board[m.old_x][m.old_y] in KINGS
                        and m.old_x == m.new_x + 3):
                        m.execute()
                    elif (move[-2] == '=' and m.promotion_piece and
                          move[-1] == m.promotion_piece.upper() and
                          move[-4:-2] == new_square and move[0] == chr(m.old_x + 97)):
                        m.execute() # like e8=Q or dxe8=Q
                    elif len(move) == 2:
                        if (move[:2] == new_square and g.board[m.old_x][m.old_y] in PAWNS):
                            m.execute() # like e4
                    elif move[-2:] == new_square:
                        first_segment = move.index('x') if 'x' in move else len(move) - 2
                        if move[0] == chr(m.old_x + 97):
                            m.execute() # like axb4
                        elif move[0] == g.board[m.old_x][m.old_y].upper():
                            if all(ord(move[i]) in [m.old_x + 97, m.old_y + 49]
                                   for i in range(1, first_segment)):
                                m.execute() # like Rb8 or Rab8 or R6b8
        except Exception as e:
            print('exception!', e)
            print(g)
            pass
        return g # incomplete, needs piece disambiguation and promotions


    def __str__(self):
        rtn = ''
        for row in reversed(list(zip(*self.board))):
            rtn += ' | '.join('.' if x == FREE else x for x in row) + '\n'
        return rtn

    def __eq__(self, other):
        return str(self) == str(other)

class Move():
    def __init__(self, game, old_x, old_y, new_x, new_y, promotion_piece=None):
        self.game = game
        self.old_x = old_x
        self.old_y = old_y
        self.new_x = new_x
        self.new_y = new_y
        self.promotion_piece = promotion_piece
        self.source_piece = game.board[old_x][old_y]
        self.target_piece = game.board[new_x][new_y]

    def execute(self):
        colour = ord(self.source_piece) > 97
        target_square = (self.new_x, self.new_y)

        # remove target_piece and implications
        if (self.source_piece in PAWNS and target_square == self.game.en_passant):
            self.target_piece = self.game.board[self.new_x][self.old_y]
            self.game.pieces[self.target_piece].remove(
                (self.new_x, self.old_y, ord(self.target_piece) > 97))
            self.game.board[self.new_x][self.old_y] = FREE
        elif target_square in self.game.castling_trail:
            self.game.winner = colour
        elif self.target_piece != FREE:
            self.game.pieces[self.target_piece].remove(
                (self.new_x, self.new_y, ord(self.target_piece) > 97))
            if self.target_piece in KINGS:
                self.game.winner = 1 - (ord(self.target_piece) > 97)
            elif (self.target_piece in ROOKS and target_square in CORNERS):
                ind = (self.new_y // 7) * 2 + (self.new_x // 7)
                self.game.castling = self.game.castling.replace('QKqk'[ind], '')

        # move source_piece and implications
        self.game.en_passant = None
        self.game.castling_trail = []

        self.game.board[self.old_x][self.old_y] = FREE
        self.game.pieces[self.source_piece].remove((self.old_x, self.old_y, colour))

        if self.promotion_piece:
            self.game.board[self.new_x][self.new_y] = self.promotion_piece
            self.game.pieces[self.promotion_piece].append((self.new_x, self.new_y, colour))
        else:
            self.game.board[self.new_x][self.new_y] = self.source_piece
            self.game.pieces[self.source_piece].append((self.new_x, self.new_y, colour))

        if self.source_piece in PAWNS and abs(self.new_y - self.old_y) == 2:
            self.game.en_passant = (self.old_x, (self.old_y + self.new_y) // 2)
        elif self.source_piece in KINGS:
            if self.new_x == self.old_x - 2:
                self.game.castling_trail = [(self.old_x - 1, self.old_y), (self.old_x, self.old_y)]
                rook_piece = self.game.board[0][self.old_y]
                self.game.board[3][self.old_y] = rook_piece
                self.game.board[0][self.old_y] = FREE
                self.game.pieces[rook_piece].remove((0, self.old_y, colour))
                self.game.pieces[rook_piece].append((3, self.old_y, colour))
            elif self.new_x == self.old_x + 2:
                self.game.castling_trail = [(self.old_x, self.old_y), (self.old_x + 1, self.old_y)]
                rook_piece = self.game.board[7][self.old_y]
                self.game.board[5][self.old_y] = rook_piece
                self.game.board[7][self.old_y] = FREE
                self.game.pieces[rook_piece].remove((7, self.old_y, colour))
                self.game.pieces[rook_piece].append((5, self.old_y, colour))

            self.game.castling = (self.game.castling
                                  .replace(chr(75 + 32 * colour), '')
                                  .replace(chr(81 + 32 * colour), ''))
        elif self.source_piece in ROOKS and (self.old_x, self.old_y) in CORNERS:
            ind = (self.old_y // 7) * 2 + (self.old_x // 7)
            self.game.castling = self.game.castling.replace('QKqk'[ind], '')

        self.game.turn += 1

    @staticmethod
    def from_string(s, game):
        old_x, old_y = str2sq(s[:2])
        new_x, new_y = str2sq(s[2:4])
        return Move(game, old_x, old_y, new_x, new_y, s[4:] or None)

    def to_string(self):
        return sq2str(self.old_x, self.old_y) + sq2str(self.new_x, self.new_y)

    def to_SAN(self): # incomplete "short algebraic notation"
        piece_letter = self.game.board[self.old_x][self.old_y]
        rtn = ''
        if piece_letter in KINGS and abs(self.old_x - self.new_x) > 1:
            if self.old_x < self.new_x:
                return 'O-O'
            else:
                return 'O-O-O'
        elif piece_letter not in PAWNS:
            rtn += piece_letter.upper()
            for other in self.game.moves():
                other_piece = self.game.board[other.old_x][other.old_y]
                if (other_piece == piece_letter and
                    (other.new_x, other.new_y) == (self.new_x, self.new_y) and
                    other != self): # needs extra disambiguation for e.g. 3 rooks
                    if other.old_x != self.old_x:
                        rtn += sq2str(self.old_x, self.old_y)[0]
                    elif other.old_y != self.old_y:
                        rtn += sq2str(self.old_x, self.old_y)[1]
                    break
        if self.game.board[self.new_x][self.new_y] != FREE:
            if piece_letter in PAWNS:
                rtn += chr(self.old_x + 97)
            rtn += 'x'
        rtn += self.to_string()[2:]
        if self.promotion_piece:
            rtn += '=%s' % self.promotion_piece.upper()
        return rtn

    def __eq__(self, move):
        return (move.old_x == self.old_x and
                move.old_y == self.old_y and
                move.new_x == self.new_x and
                move.new_y == self.new_y and
                move.promotion_piece == self.promotion_piece)

    def __ne__(self, move):
        return not self.__eq__(move)

    def __str__(self):
        return 'Move: %s to %s%d' % (self.source_piece, chr(self.new_x + 65), self.new_y + 1)

def sq2str(x, y):
    return chr(x + 97) + str(y + 1)

def str2sq(s):
    return (ord(s[0]) - 97, int(s[1]) - 1)

def simulate(move):
    new_move = copy.copy(move)
    new_move.game = Game(move.game)
    new_move.execute()
    return new_move.game
