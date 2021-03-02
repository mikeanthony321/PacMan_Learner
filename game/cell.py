from coin import *

class CellMap:
    def __init__(self):
        self.map: Cell = self.getCells()

    def getCells(self):
        cells: Cell = []
        lineCount = 0
        rowCount = 0
        walls = open("db/walls.txt", "r")

        for line in walls:
            if lineCount == 28:
                lineCount = 0
                rowCount += 1
            if "row" not in line:
                # remove \n off end of string
                line = line.strip()
                row = line.split(", ")
                for cell in row:
                    cells.append(Cell(cell, (lineCount, rowCount)))
                    lineCount += 1
        return cells

    def getCell(self, pos):
        for cell in self.map:
            if cell.pos == pos:
                return cell

    def detectCollision(self, pos):
        if self.getCell(pos).hasWall == True:
            return True
        else:
            return False

    def collectCoin(self, pos):
        cell = self.getCell(pos)
        if cell.hasCoin:
            cell.hasCoin = False
            return cell.coin.score
        else:
            return 0

class Cell:
    def __init__(self, hasWall, pos):
        self.hasWall = self.toBool(hasWall)
        self.pos = pos
        self.hasCoin = not self.hasWall  # essentially, if no collision, spawn coin

        if self.hasCoin:
            self.coin = Coin()
            if pos == (6, 8) or pos == (21, 8) or pos == (6, 20) or pos == (21, 20):
                self.coin = SuperCoin()

    def toBool(self, s):
        if s == '1':
            return True
        elif s == '0':
            return False
        else:
            raise ValueError
