# interface settings
WIDTH, HEIGHT = 560, 715
PAD_TOP, PAD_BOT = 25, 50
GRID_PIXEL_H = HEIGHT - PAD_TOP - PAD_BOT
GRID_W, GRID_H = 28, 32
CELL_W, CELL_H = WIDTH//GRID_W, GRID_PIXEL_H//GRID_H
FPS = 60

# db
HIGH_SCORE = ("HIGH SCORE" + open("db/hs.txt", "r").read())

# color settings
BLACK = (0,0,0,)
WHITE = (255,255,255)
GOLD = (255,189,51)
CERU = (51,199,255)

# font settings
TITLE_TEXT_SIZE = 16
TITLE_FONT = "barcade.ttf"