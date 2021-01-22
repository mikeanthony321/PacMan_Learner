import numpy as np


class CellMap:
    def __init__(self):
        self.map = self.getCells()

    def getCells(self):
        cells = np.empty(shape=(28, 32), dtype=object)
        lineCount = 0
        rowCount = 0

        walls = open("db/walls.txt", "r")

        for line in walls:
            if lineCount == 28:
                lineCount = 0
                rowCount += 1
            if "row" not in line:
                lineCount += 1
                # remove \n off end of string
                line = line.strip()
                # cellmap entry format: "bool, bool, bool, bool"
                cells[lineCount - 1, rowCount] = Cell(line.split(", ", 4))
        return cells

    def getCell(self, pos):
        return self.map[pos]

    def detectCollision(self, pos):
        if self.getCell(pos).leftWall == True:
            return True
        else:
            return False

class Cell:
    # todo: simplify the cells, they really don't need four walls, just one collision switch
    def __init__(self, cellData):
        self.leftWall = self.toBool(cellData[0])
        self.rightWall = self.toBool(cellData[1])
        self.topWall = self.toBool(cellData[2])
        self.bottomWall = self.toBool(cellData[3])

    def toBool(self, s):
        if s == '1':
            return True
        elif s == '0':
            return False
        else:
            raise ValueError