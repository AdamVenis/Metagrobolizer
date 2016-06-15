from game import *

class MinMaxAgent():
    def __init__(self, depth=3):
        self.depth = depth

    def move(self, game):
        results = [(-10**6 * (-1)**i, None) for i in range(self.depth + 1)]
        nodes = [[(simulate(move), [move]) for move in game.moves()]]
        eval_cache = {}
        ehits = 0
        while nodes:
            # expand(nodes)
            while len(nodes) < self.depth and nodes[-1]:
                g, m = nodes[-1].pop()
                nodes.append(self.sort_moves([(simulate(move), m + [move]) for move in g.moves()]))
            flag = True
            while flag:
                # evaluate deepest node
                if not nodes[-1]: # no moves available
                    if g.winner == -1:
                        results[len(nodes)] = (0, m)
                    else:
                        results[len(nodes)] = ((1000 - len(nodes))*(-1)**(game.turn + g.winner), m)
                else:
                    g, m = nodes[-1].pop()
                    key = str(g.board)
                    if key not in eval_cache:
                        eval_cache[key] = self.static_eval(g)*(-1)**game.turn
                    else:
                        ehits += 1
                    results[-1] = (eval_cache[key], m)

                # update running results
                i = len(nodes) - 1
                while i >= 0:
                    minmax = max if i % 2 == 0 else min
                    results[i] = minmax(results[i], results[i + 1], key=itemgetter(0))
                    if i > 0 and results[i] == minmax(results[i], results[i-1], key=itemgetter(0)):
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

    def sort_moves(self, moves):
        # most important moves should be last, since they get popped from the end
        priorities = {-1: -1, 'p': 1, 'P': 1, 'n': 3, 'N': 3,
                      'b': 3, 'B': 3, 'r': 5, 'R': 5,
                      'q': 9, 'Q': 9, 'k': 1000, 'K': 1000}
        return sorted(moves, key=lambda x: priorities[x[1][-1].target_piece])

    def static_eval(self, game):
        if game.winner == WHITE:
            return 1000
        elif game.winner == BLACK:
            return -1000
        else:
            return self.eval_player(game, WHITE) - self.eval_player(game, BLACK)

    def eval_player(self, game, colour):
        rtn = 0
        for piece, posns in game.pieces.items():
            if colour == (ord(piece) > 97):
                rtn += PIECE_VALS[piece] * len(posns)
                for x, y, c in posns:
                    for move in PIECE_FNS[piece](game, x, y, c):
                        rtn += 0.1
                        if game.board[move.new_x][move.new_y] != FREE: # wrong for promotions
                            rtn += 0.2
        return rtn
