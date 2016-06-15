
class SelectiveSearchAgent():
    def __init__(self, duration=None, depth=3):
        self.duration = duration
        self.depth = depth

    def move(self, game):
        return game.moves()[0]

    def 