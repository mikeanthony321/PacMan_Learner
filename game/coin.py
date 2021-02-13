class Coin:
    def __init__(self):
        self.score = 10
        self.isSuperCoin = False


class SuperCoin(Coin):
    def __init__(self):
        self.score = 15
        self.isSuperCoin = True
