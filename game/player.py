import pygame
from settings import *

vec = pygame.math.Vector2


class Player:
    def __init__(self, game, pos):
        self.game = game
        self.grid_pos = pos
        self.pixel_pos = vec(self.grid_pos.x * CELL_W + (CELL_W // 2),
                             self.grid_pos.y * CELL_H + (CELL_H // 2) + PAD_TOP)
        self.direction = vec(1, 0) # pacman must spawn in already moving
        self.memory_direction = None

    def update(self):
        # collision detection
        coords = (int(self.grid_pos[0] + self.direction.x), int(self.grid_pos[1] + self.direction.y))
        if self.game.cells.detectCollision(coords):
            self.direction = self.direction * -1

        # player movement
        self.pixel_pos += self.direction
        if (self.pixel_pos.x - 30) % CELL_W == 0:
            self.grid_pos[0] = (self.pixel_pos.x - CELL_W // 2) // CELL_W

            if self.direction.x != 0:
                if self.memory_direction is not None:
                    self.direction = self.memory_direction


        if (self.pixel_pos.y - 55) % CELL_H == 0:
            self.grid_pos[1] = (self.pixel_pos.y - (CELL_H // 2) - PAD_TOP) // CELL_H

            if self.direction.y != 0:
                if self.memory_direction is not None:
                    self.direction = self.memory_direction

    def draw(self):
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
            pygame.draw.rect(self.game.screen, CERU,
                             (self.grid_pos[0] * CELL_W, self.grid_pos[1] * CELL_H + PAD_TOP, CELL_W, CELL_H), 2)
            # tested cell
            pygame.draw.rect(self.game.screen, WHITE,
                             ((self.grid_pos[0] + self.direction.x) * CELL_W,
                              (self.grid_pos[1] + self.direction.y) * CELL_H + PAD_TOP, CELL_W, CELL_H), 2)
            
    def move(self, direction):
        self.memory_direction = direction

    # for ai
    def moveLeft(self):
        self.move(vec(1, 0))

    def moveRight(self):
        self.move(vec(-1, 0))

    def moveUp(self):
        self.move(vec(0, -1))

    def moveDown(self):
        self.move(vec(0, 1))
