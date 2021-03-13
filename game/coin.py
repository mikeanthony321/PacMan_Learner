from game.settings import *

class Coin:
    def __init__(self):
        self.score = COIN_SCORE
        self.isSuperCoin = False


class SuperCoin(Coin):
    def __init__(self):
        self.score = SUPERCOIN_SCORE
        self.isSuperCoin = True
