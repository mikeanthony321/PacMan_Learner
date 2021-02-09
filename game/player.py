from settings import *

vec = pygame.math.Vector2


def isAlignedX(pixel):
    return (pixel.x - 30) % CELL_W == 0


def isAlignedY(pixel):
    return (pixel.y - 55) % CELL_H == 0


class Player:
    def __init__(self, game, pos):
        # Game isntance
        self.game = game

        # 2D vector (x, y) | Pacman's current cell
        self.grid_pos = pos

        # 2D vector (x, y) | Pacman's current pixel position
        self.pixel_pos = vec(self.grid_pos.x * CELL_W + (CELL_W // 2),
                             self.grid_pos.y * CELL_H + (CELL_H // 2) + PAD_TOP)

        # The current xy direction pacman is moving
        self.direction = vec(1, 0)  # pacman must spawn in already moving

        # To prevent cell clipping, movement is only enabled during certain pixel positions.
        # Inputs are stored in this variable until direction change is allowed.
        self.requested_direction = vec(0, 0)  # initializes on first movement request
        self.canMove = True

        self.score = 0
        self.deaths = PLAYER_DEATHS

    def update(self):
        # collision detection

        # direction change request detection
        # get cell in front of requested direction
        if self.requested_direction != vec(0, 0):
            req_coords = (
                int(self.grid_pos[0] + self.requested_direction.x), int(self.grid_pos[1] + self.requested_direction.y))
            if not self.game.cells.detectCollision(req_coords):
                req_pixel = self.pixel_pos + self.requested_direction
                if self.isAlignedAndMoving(req_pixel):
                    self.direction = self.requested_direction
                    self.requested_direction = vec(0, 0)

        # get cell in front of pacman
        next_coords = (int(self.grid_pos[0] + self.direction.x), int(self.grid_pos[1] + self.direction.y))
        if not self.isValidPos(next_coords):
            next_coords = (int(self.grid_pos[0]), int(self.grid_pos[1]))

        # stop movement if next block causes collision
        if self.game.cells.detectCollision(next_coords):
            # self.direction = vec(0, 0) # this may stop you at a weird pixel pos
            self.hitWall()

        next_pixel = self.pixel_pos + self.direction

        if self.isValidPos(next_coords):
            # movement update is executed
            self.grid_pos[0] = next_pixel.x // CELL_W
            self.grid_pos[1] = (next_pixel.y - PAD_TOP) // CELL_H
            self.pixel_pos = next_pixel

            # coin mgmt
            self.score += self.game.cells.collectCoin(next_coords)

    def draw(self):
        pygame.draw.circle(self.game.screen, WHITE, (int(self.pixel_pos.x), int(self.pixel_pos.y)), 2)
        # pacman
        pygame.draw.circle(self.game.screen,
                           YELLOW,
                           (int(self.pixel_pos.x), int(self.pixel_pos.y)),
                           CELL_W // 2 - 2)
        pygame.draw.circle(self.game.screen,
                           BLACK,
                           (int(self.pixel_pos.x) + 5, int(self.pixel_pos.y)),
                           2)
        if SHOW_GRID:
            # hit box
            pygame.draw.rect(self.game.screen, BLUE,
                             (self.grid_pos[0] * CELL_W, self.grid_pos[1] * CELL_H + PAD_TOP, CELL_W, CELL_H), 2)
            # tested cell
            pygame.draw.rect(self.game.screen, WHITE,
                             ((self.grid_pos[0] + self.direction.x) * CELL_W,
                              (self.grid_pos[1] + self.direction.y) * CELL_H + PAD_TOP, CELL_W, CELL_H), 2)

            # requested cell
            if self.requested_direction is not None:
                pygame.draw.rect(self.game.screen, RED,
                    ((self.grid_pos[0] + self.requested_direction.x) * CELL_W,
                    (self.grid_pos[1] + self.requested_direction.y) * CELL_H + PAD_TOP, CELL_W, CELL_H), 2)

    def reset(self):
        self.direction = vec(0, 0)

    def teleport(self):
        self.reset()
        self.grid_pos = vec(1, 1)
        self.pixel_pos = vec(self.grid_pos.x * CELL_W + (CELL_W // 2),
                             self.grid_pos.y * CELL_H + (CELL_H // 2) + PAD_TOP)
        self.direction = vec(1, 0)
        # if direction != None:
        #    self.move(direction)
        print("you should be teleporting!")

    def isValidPos(self, pos):
        # return not pos[0] < 0 or pos[0] > 27 or pos[1] < 0 or pos[1] > 30
        return self.game.cells.getCell(pos) is not None

    def isAlignedAndMoving(self, pixel):
        # clipping prevention, this checks a pixel position alignment with the maze
        # could be simplified
        switch = False
        if (pixel.x - 30) % CELL_W == 0:
            # if x axis is aligned ^
            if self.requested_direction.y != 0:
                # if player requested x movement ^
                switch = True
        if (pixel.y - 55) % CELL_H == 0:
            if self.requested_direction.x != 0:
                switch = True
        return switch

    def hitWall(self):
        if self.direction.x != 0:
            if isAlignedX(self.pixel_pos):
                self.direction = vec(0, 0)
        if self.direction.y != 0:
            if isAlignedY(self.pixel_pos):
                self.direction = vec(0, 0)

    def move(self, direction):
        self.requested_direction = direction

        if not self.canMove:
            self.direction = direction

    def getGridPos(self, pos):
        pos -= vec(10, 35)
        # from x = 10 to x = 29, formula should output 0
        # from x = 30 to x = 49, formular should output 1

        return grid
