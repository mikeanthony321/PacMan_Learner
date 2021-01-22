import pygame
from settings import *

vec = pygame.math.Vector2

class Player:
    def __init__(self, game, pos):
        self.game = game
        self.grid_pos = pos
        self.pixel_pos = vec(self.grid_pos.x * CELL_W + CELL_W//2, self.grid_pos.y * CELL_H + CELL_H//2 + PAD_TOP)
        self.direction = vec(0,0)

    def update(self):
        # collision detection
        nextCell = self.game.cells.getCell((int(self.grid_pos[0] + self.direction.x), int(self.grid_pos[1] + self.direction.y)))
        if(self.direction.x == 1 & nextCell.rightWall == True):
            self.direction = vec(0,0)
        if(self.direction.x == -1 & nextCell.leftWall == True):
            self.direction = vec(0,0)
        if(self.direction.y == 1 & nextCell.bottomWall == True):
            self.direction = vec(0,0)
        if(self.direction.y == -1 & nextCell.topWall == True):
            self.direction = vec(0,0)

        # player movement
        self.pixel_pos += self.direction
        self.grid_pos = ((self.pixel_pos[0] + PAD_TOP)//CELL_W - 1, (self.pixel_pos[1]//CELL_H) - 1)

    def draw(self):
        # pacman
        pygame.draw.circle(self.game.screen,
                           YELLOW,
                           (int(self.pixel_pos.x), int(self.pixel_pos.y)),
                           CELL_W//2 - 2)
        pygame.draw.circle(self.game.screen  ,
                           BLACK,
                           (int(self.pixel_pos.x) + 5, int(self.pixel_pos.y)),
                           2)
        # hitbox
        pygame.draw.rect(self.game.screen, CERU, (self.grid_pos[0] * CELL_W, self.grid_pos[1] * CELL_H + PAD_TOP, CELL_W, CELL_H), 2)

    def move(self, direction):
        self.direction = direction