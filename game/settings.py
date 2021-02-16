import pygame
vec = pygame.math.Vector2

# interface settings
WIDTH, HEIGHT = 560, 715
PAD_TOP, PAD_BOT = 25, 50
GRID_PIXEL_H = HEIGHT - PAD_TOP - PAD_BOT
GRID_W, GRID_H = 28, 32
CELL_W, CELL_H = WIDTH//GRID_W, GRID_PIXEL_H//GRID_H
FPS = 60
SHOW_GRID = True

# db
HIGH_SCORE: int = open("db/hs.txt", "r").readline()

# color settings
BLACK = (0,0,0,)
WHITE = (255,255,255)
GOLD = (255,189,51)
CERU = (51,199,255)
YELLOW = (190, 190, 5)
RED = (255, 0, 0)

# font settings
TITLE_TEXT_SIZE = 16
TITLE_FONT = "res/barcade.ttf"

# player settings
PLAYER_START_POS = vec(1, 1)
PLAYER_DEATHS = open("db/deaths.txt", "r").readline()

# ghost settings
INKY_START_POS = vec(12, 11)
BLINKY_START_POS = vec(15, 11)
PINKY_START_POS = vec(12, 17)
CLYDE_START_POS = vec(15, 17)

# spritesheet settings
SPRITE_SIZE = 16

MOVE_RIGHT = 1
MOVE_LEFT = 2
MOVE_UP = 3
MOVE_DOWN = 4

BLINKY_SPRITE_POS = 2
PINKY_SPRITE_POS = 3
INKY_SPRITE_POS = 4
CLYDE_SPRITE_POS = 5