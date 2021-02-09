import pygame

vec = pygame.math.Vector2

# interface settings
WIDTH, HEIGHT = 560, 715
PAD_TOP, PAD_BOT = 25, 50
GRID_PIXEL_H = HEIGHT - PAD_TOP - PAD_BOT
GRID_W, GRID_H = 28, 32
CELL_W, CELL_H = WIDTH // GRID_W, GRID_PIXEL_H // GRID_H
FPS = 60
SHOW_GRID = True

# db
HIGH_SCORE: int = open("db/hs.txt", "r").readline()

# color settings
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GOLD = (255, 189, 51)
BLUE = (51, 199, 255)
YELLOW = (190, 190, 5)
RED = (255, 0, 0)

# font settings
TITLE_TEXT_SIZE = 16
TITLE_FONT = "barcade.ttf"

# player settings
PLAYER_START_POS = vec(1, 1)
PLAYER_DEATHS = open("db/deaths.txt", "r").readline()
