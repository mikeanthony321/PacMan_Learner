import random
from settings import *
vec = pygame.math.Vector2

class Ghost(pygame.sprite.Sprite):
    def __init__(self, game, screen, aggressive, name, pos, sprite_pos, sprite_sheet):
        super().__init__()
        # Game / rendering context
        self.game = game
        self.screen = screen

        # Ghost Information
        self.aggressive = aggressive
        self.name = name
        self.active = True
        self.should_display = True

        # Vector indicating the Cell of the Grid currently occupied by the Ghost
        self.grid_pos = pos

        # pixel position of center of ghost
        self.pixel_pos = vec(self.grid_pos.x * CELL_W + (CELL_W // 2),
                             self.grid_pos.y * CELL_H + (CELL_H // 2) + PAD_TOP)
        # pixel position for top left corner of image surface
        self.image_pos = vec(self.grid_pos.x * CELL_W,
                             self.grid_pos.y * CELL_H + PAD_TOP)

        # Currently, all four ghosts begin with the same direction, this can be randomized
        self.direction = vec(-1, 0)
        self.speed = 1.0

        # Sprite Sheet information
        self.sprite_pos = sprite_pos
        self.sprite_sheet = sprite_sheet
        self.frames = []
        self.getFrames()
        self.frame = self.frames[0]
        self.frame_count = 0
        self.frame_changes = 0

    def getFrames(self):
        # Store the animation frames for the Ghost within a list
        for x in range(0, 8):
            image = pygame.Surface([SPRITE_SIZE,SPRITE_SIZE])
            image.blit(self.sprite_sheet,
                       (0, 0),
                       (x * SPRITE_SIZE, self.sprite_pos * SPRITE_SIZE, SPRITE_SIZE, SPRITE_SIZE))
            image.set_colorkey(BLACK)
            image = pygame.transform.scale(image, (CELL_W, CELL_H))
            self.frames.append(image)

    def updateFrame(self):

        # update normal condition ghost animation frames
        if self.direction.x == -1:
            self.frame = self.frames[self.frame_changes + GHOST_LEFT]
        elif self.direction.x == 1:
            self.frame = self.frames[self.frame_changes + GHOST_RIGHT]
        elif self.direction.y == -1:
            self.frame = self.frames[self.frame_changes + GHOST_UP]
        elif self.direction.y == 1:
            self.frame = self.frames[self.frame_changes + GHOST_DOWN]
        self.frame_changes += 1
        if self.frame_changes == 2:
            self.frame_changes = 0

        # todo: implement animation frames for ghosts during Power pellet
        # todo: implement animation frames for ghosts consumed by Pac-Man


    def checkTile(self):
        # initialize an array to hold possible direction changes
        poss_dir = []

        # check tiles above and below
        if GRID[int(self.grid_pos.y - 1)][int(self.grid_pos.x)] == 1:
            poss_dir.append(vec(0, -1))
        if GRID[int(self.grid_pos.y + 1)][int(self.grid_pos.x)] == 1:
            poss_dir.append(vec(0, 1))

        # check tiles to the left and right
        if GRID[int(self.grid_pos.y)][int(self.grid_pos.x - 1)] == 1:
            poss_dir.append(vec(-1, 0))
        if GRID[int(self.grid_pos.y)][int(self.grid_pos.x + 1)] == 1:
            poss_dir.append(vec(1, 0))

        # choose a direction randomly from available changes
        if len(poss_dir) != 0:
            rand = random.randint(0, len(poss_dir) - 1)
            if poss_dir[rand] != -self.direction:
                self.direction = poss_dir[rand]
                # call for a frame update regardless of 15 fps animations
                self.updateFrame()

    def update(self):
        #15 frames for ghost frame changes seems about right, though increased speed may make it look weird
        self.frame_count += 1

        if self.frame_count % 15 == 0:
            self.updateFrame()

        # collision detection
        coords = (int(self.grid_pos[0] + self.direction.x), int(self.grid_pos[1] + self.direction.y))
        if self.game.cells.detectCollision(coords):
            self.checkTile()


        # movement
        if self.active:
            self.pixel_pos += (self.direction * self.speed)
            self.image_pos += (self.direction * self.speed)

        if self.direction.x != 0:
            # Check if the ghost has entered a new grid tile and update direction as necessary
            if (self.pixel_pos.x - 30) % CELL_W == 0:
                self.grid_pos[0] = (self.pixel_pos.x - CELL_W // 2) // CELL_W
                self.checkTile()

        if self.direction.y != 0:
            # Check if the ghost has entered a new grid tile and update direction as necessary
            if (self.pixel_pos.y - 55) % CELL_H == 0:
                self.grid_pos[1] = (self.pixel_pos.y - (CELL_H // 2) - PAD_TOP) // CELL_H
                self.checkTile()


    def draw(self):
        # Draw the ghost to the pygame screen
        if self.should_display:
            self.screen.blit(self.frame, self.image_pos)

    # Currently increasing the speed creates the potential to overshoot collision detection and will need to be
    # tweaked if we wish to implement it.
    def setSpeed(self, new_speed):
        self.speed = new_speed

    def getPixelPos(self):
        return self.pixel_pos

    def setActiveStatus(self, status):
        self.active = status

    def setDisplayStatus(self, status):
        self.should_display = status