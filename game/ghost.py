from settings import *
vec = pygame.math.Vector2

class Ghost:
    def __init__(self, game, aggressive, name, pos, sprite):
        self.game = game
        self.aggressive = aggressive
        self.name = name
        self.grid_pos = pos
        self.pixel_pos = vec(self.grid_pos.x * CELL_W,
                             self.grid_pos.y * CELL_H + PAD_TOP)
        # In the classic, only one ghost begins moving right away, the rest move one after the other after a few seconds
        self.direction = vec(0, 0)
        self.sprite = sprite
        self.sprite = pygame.transform.scale(self.sprite, (CELL_W, CELL_H))

    def update(self):
        # collision detection
        coords = (int(self.grid_pos[0] + self.direction.x), int(self.grid_pos[1] + self.direction.y))
        if self.game.cells.detectCollision(coords):
            self.direction = self.direction * -1

    def draw(self):
        self.game.blit(self.sprite, self.pixel_pos)
