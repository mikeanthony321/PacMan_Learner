from settings import *
vec = pygame.math.Vector2

class Player:
    def __init__(self, game, screen, pos, sprite_sheet):
        self.game = game
        self.screen = screen
        self.grid_pos = pos

        # pixel position of center of Pac-Man
        self.pixel_pos = vec(self.grid_pos.x * CELL_W + (CELL_W // 2),
                             self.grid_pos.y * CELL_H + (CELL_H // 2) + PAD_TOP)

        # pixel position for top left corner of image surface
        self.sprite_pos = vec(self.grid_pos.x * CELL_W,
                              self.grid_pos.y * CELL_H + PAD_TOP)
        self.direction = vec(1, 0) # pacman must spawn in already moving
        self.memory_direction = None

        # player metrics
        self.score = 0
        self.deaths = PLAYER_DEATHS

        # Game status
        self.alive = True
        self.game_over = False

        # Sprite Sheet information
        self.sprite_sheet = sprite_sheet
        self.frames = []
        self.death_frames = []
        self.getFrames()
        self.frame = self.frames[0]
        self.frame_count = 0
        self.frame_changes = 0

    def getFrames(self):
        # Full Pac-Man frame
        image = pygame.Surface([SPRITE_SIZE, SPRITE_SIZE])
        image.blit(self.sprite_sheet,
                   (0,0),
                   (0, SPRITE_SIZE, SPRITE_SIZE, SPRITE_SIZE))
        image.set_colorkey(BLACK)
        image = pygame.transform.scale(image, (CELL_W, CELL_H))
        self.frames.append(image)

        # Directional frames for Pac-Man
        for x in range(0, 8):
            image = pygame.Surface([SPRITE_SIZE, SPRITE_SIZE])
            image.blit(self.sprite_sheet,
                       (0,0),
                       (x * SPRITE_SIZE, 0, SPRITE_SIZE, SPRITE_SIZE))
            image.set_colorkey(BLACK)
            image = pygame.transform.scale(image, (CELL_W, CELL_H))
            self.frames.append(image)

        # Death animation frames
        self.death_frames.append(self.frames[0]) #first frame is same as stationary frame
        for x in range(1, 11):
            image = pygame.Surface([SPRITE_SIZE, SPRITE_SIZE])
            image.blit(self.sprite_sheet,
                       (0,0),
                       (x * SPRITE_SIZE, SPRITE_SIZE, SPRITE_SIZE, SPRITE_SIZE))
            image.set_colorkey(BLACK)
            image = pygame.transform.scale(image, (CELL_W, CELL_H))
            self.death_frames.append(image)

    def updateFrame(self, frames):
        # update frames for Pac-Man while alive
        if self.alive:
            self.frame_changes += 1
            if self.direction.x == 1:
                frame_number = 5 % self.frame_changes
                self.frame = frames[frame_number]
            elif self.direction.x == -1:
                frame_number = (5 % self.frame_changes)
                if frame_number != 0:
                    frame_number += 2
                self.frame = frames[frame_number]
            elif self.direction.y == 1:
                frame_number = (5 % self.frame_changes)
                if frame_number != 0:
                    frame_number += 4
                self.frame = frames[frame_number]
            elif self.direction.y == -1:
                frame_number = (5 % self.frame_changes)
                if frame_number != 0:
                    frame_number += 6
                self.frame = frames[frame_number]
            if self.frame_changes == 5:
                self.frame_changes = 1

        # run through death frames list to animate Pac-Man death
        if not self.alive and self.frame_changes < len(self.death_frames):
            self.frame = self.death_frames[self.frame_changes]
            self.frame_changes += 1

    def update(self):
        self.frame_count += 1

        if self.direction.x == 0 and self.direction.y == 0:
            self.frame = self.frames[0]
        if self.frame_count % 6 == 0:
            if self.alive:
                self.updateFrame(self.frames)
            else:
                self.updateFrame(self.death_frames)

        # collision detection
        coords = (int(self.grid_pos[0] + self.direction.x), int(self.grid_pos[1] + self.direction.y))
        if self.game.cells.detectCollision(coords):
            self.direction = self.direction * -1

        # player movement
        if self.alive:
            self.pixel_pos += self.direction
            self.sprite_pos += self.direction
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

        # coin mgmt
        self.score += self.game.cells.collectCoin(coords)

    def draw(self):
        self.screen.blit(self.frame, self.sprite_pos)
        # pacman
        #pygame.draw.circle(self.game.screen,
        #                   YELLOW,
        #                   (int(self.pixel_pos.x), int(self.pixel_pos.y)),
        #                   CELL_W // 2 - 2)
        #pygame.draw.circle(self.game.screen,
        #                   BLACK,
        #                   (int(self.pixel_pos.x) + 5, int(self.pixel_pos.y)),
        #                   2)
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

    def getPixelPos(self):
        return self.pixel_pos

    def getAliveStatus(self):
        return self.alive

    def setAliveStatus(self, status):
        self.alive = status
        self.frame_changes = 0