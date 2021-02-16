import random
from settings import *
vec = pygame.math.Vector2

class Ghost(pygame.sprite.Sprite):
    def __init__(self, game, screen, aggressive, name, pos, sprite_pos, sprite_sheet):
        super().__init__()
        self.game = game
        self.screen = screen
        self.aggressive = aggressive
        self.name = name
        self.grid_pos = pos
        self.pixel_pos = vec(self.grid_pos.x * CELL_W + (CELL_W // 2),
                             self.grid_pos.y * CELL_H + (CELL_H // 2) + PAD_TOP)
        self.image_pos = vec(self.grid_pos.x * CELL_W,
                             self.grid_pos.y * CELL_H + PAD_TOP)
        # In the classic, only one ghost begins moving right away, the rest move one after the other after a few seconds
        self.direction = vec(0, 0)
        self.memory_direction = None
        # randomize an initial direction - will need to be altered to only do this if a derpy ghost
        rand_dir_x_or_y, rand_dir_pos_or_neg = random.random(), random.random()

        if rand_dir_x_or_y < 0.5:
            if rand_dir_pos_or_neg < 0.5:
                self.direction.x = -1
            else:
                self.direction.x = 1
        else:
            if rand_dir_pos_or_neg >= 0.5:
                self.direction.y = -1
            else:
                self.direction.y = 1

        print(self.direction)

        self.sprite_pos = sprite_pos
        self.sprite_sheet = sprite_sheet

        self.frames = []
        self.get_frames()

        self.frame = self.frames[0]

        self.frame_count = 0
        self.count = 0

    def get_frames(self):
        for x in range(0, 8):
            image = pygame.Surface([SPRITE_SIZE,SPRITE_SIZE])
            image.blit(self.sprite_sheet,
                       (0, 0),
                       (x * SPRITE_SIZE, self.sprite_pos * SPRITE_SIZE, SPRITE_SIZE, SPRITE_SIZE))
            image.set_colorkey(BLACK)
            image = pygame.transform.scale(image, (CELL_W, CELL_H))
            self.frames.append(image)

    def randomize_direction(self):
        if self.direction.x != 0:
            coords = (int(self.grid_pos[0]), int(self.grid_pos[1] + 1))
            y_dir = vec(0, 0)
            rand_dir_x_or_y, rand_dir_pos_or_neg = random.random(), random.random()

            if rand_dir_x_or_y >= 0.5:

                if not self.game.cells.detectCollision(coords):
                    y_dir.x += 1
                coords = (int(self.grid_pos[0]), int(self.grid_pos[1] - 1))
                if not self.game.cells.detectCollision(coords):
                    y_dir.y += 1
                if y_dir.x == 0 and y_dir.y == 0:
                    self.direction *= -1
                else:
                    if rand_dir_pos_or_neg >= 0.5 and y_dir.x > 0:
                        self.direction.x = 0
                        self.direction.y = 1
                    elif rand_dir_pos_or_neg < 0.5 and y_dir.y > 0:
                        self.direction.x = 0
                        self.direction.y = -1
            else:
                self.direction *= -1


        if self.direction.y != 0:
            coords = (int(self.grid_pos[0] + 1), int(self.grid_pos[1]))
            x_dir = vec(0, 0)
            rand_dir_x_or_y, rand_dir_pos_or_neg = random.random(), random.random()
            if rand_dir_x_or_y < 0.5:

                if not self.game.cells.detectCollision(coords):
                    x_dir.x += 1
                coords = (int(self.grid_pos[0] - 1), int(self.grid_pos[1]))
                if not self.game.cells.detectCollision(coords):
                    x_dir.y += 1
                if x_dir.x == 0 and x_dir.y == 0:
                    self.direction *= -1
                else:
                    if rand_dir_pos_or_neg >= 0.5 and x_dir.x > 0:
                        self.direction.x = 0
                        self.direction.y = 1
                    elif rand_dir_pos_or_neg < 0.5 and x_dir.y > 0:
                        self.direction.x = 0
                        self.direction.y = -1
            else:
                self.direction *= -1


    def update(self):
        #15 frames for ghost changes seems about right.
        self.frame_count += 1

        if self.frame_count % 15 == 0:
            self.count += 1
            if self.count >= 8:
                self.count = 0
        self.frame = self.frames[self.count]

        # collision detection
        coords = (int(self.grid_pos[0] + self.direction.x), int(self.grid_pos[1] + self.direction.y))
        if self.game.cells.detectCollision(coords):
            #self.direction = self.direction * -1
            if not self.aggressive:
                self.randomize_direction()


        # ghost movement
        self.pixel_pos += self.direction
        self.image_pos += self.direction
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
        self.screen.blit(self.frame, self.image_pos)
