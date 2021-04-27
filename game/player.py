from settings import *
import random

vec = pygame.math.Vector2


def isAligned(pixel):
    return isAlignedX(pixel) and isAlignedY(pixel)


def isAlignedX(pixel):
    return (pixel.x - 30) % CELL_W == 0


def isAlignedY(pixel):
    return (pixel.y - 55) % CELL_H == 0


class Player:
    def __init__(self, game, screen, pos, respawn_pos, sprite_sheet):
        # Game instance
        self.game = game

        # Game Window
        self.screen = screen

        # 2D vector (x, y) | Pacman's current cell | Pacman's presence cell
        self.grid_pos = pos
        self.respawn_pos = respawn_pos
        self.presence_pos = self.grid_pos

        # 2D vector (x, y) | Pacman's current pixel position (center)
        self.pixel_pos = vec(self.grid_pos.x * CELL_W + (CELL_W // 2),
                             self.grid_pos.y * CELL_H + (CELL_H // 2) + PAD_TOP)

        # 2D vector (x, y) | Pacman's current sprite position (top left corner)
        self.sprite_pos = vec(self.grid_pos.x * CELL_W,
                              self.grid_pos.y * CELL_H + PAD_TOP)

        # The current xy direction pacman is moving
        self.direction = vec(1, 0)  # pacman must spawn in already moving

        # To prevent cell clipping, movement is only enabled during certain pixel positions.
        # Inputs are stored in this variable until direction change is allowed.
        self.requested_direction = vec(0, 0)  # initializes on first movement request

        # player metrics
        self.score = 0
        self.deaths = int(PLAYER_DEATHS)

        # Game status
        self.alive = True
        self.game_over = False
        self.power_pellet_active = False
        self.power_pellet_time = 0

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
                   (0, 0),
                   (0, SPRITE_SIZE, SPRITE_SIZE, SPRITE_SIZE))
        image.set_colorkey(BLACK)
        image = pygame.transform.scale(image, (CELL_W, CELL_H))
        self.frames.append(image)

        # Directional frames for Pac-Man
        for x in range(0, 8):
            image = pygame.Surface([SPRITE_SIZE, SPRITE_SIZE])
            image.blit(self.sprite_sheet,
                       (0, 0),
                       (x * SPRITE_SIZE, 0, SPRITE_SIZE, SPRITE_SIZE))
            image.set_colorkey(BLACK)
            image = pygame.transform.scale(image, (CELL_W, CELL_H))
            self.frames.append(image)

        # Death animation frames
        self.death_frames.append(self.frames[0])  # first frame is same as stationary frame
        for x in range(1, 12):
            image = pygame.Surface([SPRITE_SIZE, SPRITE_SIZE])
            image.blit(self.sprite_sheet,
                       (0, 0),
                       (x * SPRITE_SIZE, SPRITE_SIZE, SPRITE_SIZE, SPRITE_SIZE))
            image.set_colorkey(BLACK)
            image = pygame.transform.scale(image, (CELL_W, CELL_H))
            self.death_frames.append(image)

    def update_frame(self, frames):
        # update frames for Pac-Man while alive
        if self.alive:
            if self.direction.x == 0 and self.direction.y == 0:
                self.frame = self.frames[0]
            self.frame_changes += 1
            if self.direction.x == 1:
                frame_number = 5 % self.frame_changes
                self.frame = frames[frame_number]
            elif self.direction.x == -1:
                frame_number = (5 % self.frame_changes)
                if frame_number != 0:
                    frame_number += 2
                self.frame = frames[frame_number]
            elif self.direction.y == -1:
                frame_number = (5 % self.frame_changes)
                if frame_number != 0:
                    frame_number += 4
                self.frame = frames[frame_number]
            elif self.direction.y == 1:
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
            if self.frame_changes == len(self.death_frames):
                self.game_over = True

    def update(self):
        self.frame_count += 1

        if self.frame_count % 6 == 0:
            if self.alive:
                self.update_frame(self.frames)
            else:
                self.update_frame(self.death_frames)

        if not self.alive and self.game_over:
            if isAligned(self.pixel_pos):
                self.reset()

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
            self.hitWall()

        next_pixel = self.pixel_pos + self.direction

        if self.isValidPos(next_coords):
            # movement update is executed
            self.grid_pos[0] = next_pixel.x // CELL_W
            self.grid_pos[1] = (next_pixel.y - PAD_TOP) // CELL_H
            self.pixel_pos = next_pixel
            self.sprite_pos += self.direction

            if isAligned(self.pixel_pos):
                # coin mgmt
                score_up = self.game.cells.collectCoin(self.grid_pos)
                # supercoin mgmt
                if score_up == SUPERCOIN_SCORE:
                    self.power_pellet_active = True
                    self.power_pellet_time = POWER_PELLET_TIMER
                self.score += score_up

        if abs(self.grid_pos.x - self.presence_pos.y) > 2 or abs(self.grid_pos.y - self.presence_pos.y) > 2:
            self.presence_pos = self.grid_pos

    def draw(self):
        # pacman
        self.screen.blit(self.frame, self.sprite_pos)

        if SHOW_GRID:
            # hit box
            pygame.draw.rect(self.game.screen, BLUE,
                             (self.grid_pos[0] * CELL_W, self.grid_pos[1] * CELL_H + PAD_TOP, CELL_W, CELL_H), 2)
            # tested cell
            pygame.draw.rect(self.game.screen, WHITE,
                             ((self.grid_pos[0] + self.direction.x) * CELL_W,
                              (self.grid_pos[1] + self.direction.y) * CELL_H + PAD_TOP, CELL_W, CELL_H), 2)

    def reset(self):
        self.deaths += 1
        self.stop()
        # This is only set up to always randomize the spawn, we can add the choice when PAC-25
        # gets fully implemented
        self.teleport(self.setRandomSpawnLocation())
        # set to vector to avoid wrong respawn bug
        # self.teleport(vec(13, 23))

    def setRandomSpawnLocation(self):
        while(True):
            xPos = random.choice([random.randint(0, 8), random.randint(19, 27)])
            yPos = random.choice([random.randint(0, 10), random.randint(18, 30)])
            if GRID[yPos][xPos] == 1:
                return vec(xPos, yPos)

    def teleport(self, pos):
        self.grid_pos = pos
        self.sprite_pos = vec(self.grid_pos.x * CELL_W,
                              self.grid_pos.y * CELL_H + PAD_TOP)
        self.pixel_pos = vec(self.grid_pos.x * CELL_W + (CELL_W // 2),
                             self.grid_pos.y * CELL_H + (CELL_H // 2) + PAD_TOP)

    def stop(self):
        self.direction = vec(0, 0)

    def isValidPos(self, pos):
        # return not pos[0] < 0 or pos[0] > 27 or pos[1] < 0 or pos[1] > 30
        return self.game.cells.getCell(pos) is not None

    def isAlignedAndMoving(self, pixel):
        # clipping prevention, this checks a pixel position alignment with the maze
        return ((pixel.x - 30) % CELL_W == 0 and self.requested_direction.y != 0) or (
                (pixel.y - 55) % CELL_H == 0 and self.requested_direction.x != 0)

    def hitWall(self):
        if (self.direction.x != 0 and isAlignedX(self.pixel_pos)) or (
                self.direction.y != 0 and isAlignedY(self.pixel_pos)):
            self.direction = vec(0, 0)

    def move(self, direction):
        self.requested_direction = direction

    def get_pixel_pos(self):
        return self.pixel_pos

    def get_grid_pos(self):
        return self.grid_pos

    def get_alive_status(self):
        return self.alive

    def set_alive_status(self, status):
        if status == False:
            self.direction = vec(0, 0)
        self.alive = status
        self.frame_changes = 0

    def get_game_over_status(self):
        return self.game_over

    def set_game_over_status(self, game_over):
        self.game_over = game_over

    def get_presence(self):
        return self.presence_pos

    def set_power_pellet_status(self, status):
        self.power_pellet_active = status

    def get_bounds(self):
        # Returns tuple of points representing Pac-Man's boundaries
        # (left, right, up, down)
        # (vec(x,y), vec(x,y), vec(x,y), vec(x,y))
        radius = SPRITE_SIZE / 2
        return (vec(self.pixel_pos.x - 5, self.pixel_pos.y), vec(self.pixel_pos.x + 5, self.pixel_pos.y), \
                vec(self.pixel_pos.x, self.pixel_pos.y - radius), vec(self.pixel_pos.x, self.pixel_pos.y + radius))

    def power_pellet_timer(self):
        if self.power_pellet_time > 0:
            self.power_pellet_time -= 1
        else:
            self.power_pellet_active = False