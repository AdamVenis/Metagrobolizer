
import random

class RandomAgent():
    def move(self, game):
        return random.choice(game.moves())