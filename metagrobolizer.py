## TODOS:
#   modify board instance instead of create new instance (test this)
#       - deemed unnecessary for now since static_eval uses 80% of the time
#   use timer module to evaluate for a certain time period
#   better static eval heuristics
#   finish rules (draw)
#   optimize piece ordering in search (for earlier alpha-beta hits)
#       - a mate in 2 should execute quickly even with depth 10 search
##

from agents.minmax import MinMaxAgent
from agents.human import HumanAgent
from game import *

import cmd
import sys
import time
    
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
        print(game.to_FEN())
        print('Time: %.4f' % (time.time()-start))
        try:
            print('Eval: %.2f' % agent.static_eval(game))
        except Exception:
            pass
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
        for move in moves:
            Move.from_string(move, self.game).execute()

    def do_go(self, args):
        engine_agent = MinMaxAgent(4)
        print('bestmove %s' % engine_agent.move(self.game).to_string())

    def do_quit(self, args):
        sys.exit()

if __name__ == '__main__':
    play(pgn_output_file='replays/replay009.pgn') # terminal interface
    #Shell().cmdloop() # uci interface
