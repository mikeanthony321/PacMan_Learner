import random
import numpy as np
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
        self.ghost_alive = True
        self.should_display = True

        # Vector indicating the Cell of the Grid currently occupied by the Ghost
        self.start_pos = pos
        self.grid_pos = pos
        self.steps = 0
        self.next_tile = self.grid_pos

        # pixel position of center of ghost
        self.pixel_pos = vec(self.grid_pos.x * CELL_W + (CELL_W // 2),
                             self.grid_pos.y * CELL_H + (CELL_H // 2) + PAD_TOP)
        # pixel position for top left corner of image surface
        self.image_pos = vec(self.grid_pos.x * CELL_W,
                             self.grid_pos.y * CELL_H + PAD_TOP)

        # Currently, all four ghosts begin with the same direction, this can be randomized
        self.direction = vec(0, 0)
        self.speed = 1.0

        # Sprite Sheet information
        self.sprite_pos = sprite_pos
        self.sprite_sheet = sprite_sheet
        self.frames = []
        self.power_pellet_frames = []
        self.ghost_death_frames = []
        self.get_frames()
        self.frame = self.frames[0]
        self.frame_count = 0
        self.frame_changes = 0

        # Game State Information
        self.power_pellet_active = False
        self.fleeing = False
        self.respawning = False
        self.pac_pos = PLAYER_START_POS

        # Find the initial path for an aggressive ghost to Pac-Man
        if self.aggressive:
            self.path = self.a_search(self.grid_pos, self.pac_pos)

    def get_frames(self):
        # Store the animation frames for the Ghost within a list
        for x in range(0, 8):
            image = pygame.Surface([SPRITE_SIZE,SPRITE_SIZE])
            image.blit(self.sprite_sheet,
                       (0, 0),
                       (x * SPRITE_SIZE, self.sprite_pos * SPRITE_SIZE, SPRITE_SIZE, SPRITE_SIZE))
            image.set_colorkey(BLACK)
            image = pygame.transform.scale(image, (CELL_W, CELL_H))
            self.frames.append(image)

        # Store the Power Pellet State Frames
        for x in range(0, 4):
            image = pygame.Surface([SPRITE_SIZE, SPRITE_SIZE])
            image.blit(self.sprite_sheet,
                       (0, 0),
                       (x * SPRITE_SIZE, 6 * SPRITE_SIZE, SPRITE_SIZE, SPRITE_SIZE))
            image.set_colorkey(BLACK)
            image = pygame.transform.scale(image, (CELL_W, CELL_H))
            self.power_pellet_frames.append(image)

        # Store the Death Frames
        for x in range(4, 8):
            image = pygame.Surface([SPRITE_SIZE, SPRITE_SIZE])
            image.blit(self.sprite_sheet,
                       (0, 0),
                       (x * SPRITE_SIZE, 6 * SPRITE_SIZE, SPRITE_SIZE, SPRITE_SIZE))
            image.set_colorkey(BLACK)
            image = pygame.transform.scale(image, (CELL_W, CELL_H))
            self.ghost_death_frames.append(image)

    def update_frame(self):
        # update normal condition ghost animation frames
        if not self.power_pellet_active and self.ghost_alive:
            if self.direction.x == -1:
                self.frame = self.frames[self.frame_changes + GHOST_LEFT]
            elif self.direction.x == 1:
                self.frame = self.frames[self.frame_changes + GHOST_RIGHT]
            elif self.direction.y == 1:
                self.frame = self.frames[self.frame_changes + GHOST_DOWN]
            elif self.direction.y == -1:
                self.frame = self.frames[self.frame_changes + GHOST_UP]
            self.frame_changes += 1
            if self.frame_changes == 2:
                self.frame_changes = 0

        # todo: implement animation frames for ghosts during Power pellet
        if self.power_pellet_active and self.ghost_alive:
            self.frame_changes = 0
            self.frame = self.power_pellet_frames[self.frame_changes]
            self.frame_changes += 1
            if self.frame_changes == 3:
                self.frame_changes = 0

        # todo: implement animation frames for ghosts consumed by Pac-Man
        if not self.ghost_alive:
            if self.direction.x == 1:
                self.frame = self.ghost_death_frames[0]
            elif self.direction.x == -1:
                self.frame = self.ghost_death_frames[1]
            elif self.direction.y == -1:
                self.frame = self.ghost_death_frames[2]
            elif self.direction.y == 1:
                self.frame = self.ghost_death_frames[3]

    def check_tile(self):
        # initialize an array to hold possible direction changes
        poss_dir = []

        if self.power_pellet_active and not self.fleeing:
            manhattan_dist = self.node_score(self.grid_pos, self.pac_pos)
            if manhattan_dist < 10:
                self.direction = -self.direction
                self.fleeing = True

        # check tiles above and below
        if GRID[int(self.grid_pos.y - 1)][int(self.grid_pos.x)] == 1:
            if -self.direction != vec(0, -1):
                poss_dir.append(vec(0, -1))

        if GRID[int(self.grid_pos.y + 1)][int(self.grid_pos.x)] == 1:
            if -self.direction != vec(0, 1):
                poss_dir.append(vec(0, 1))

        # check tiles to the left and right
        if GRID[int(self.grid_pos.y)][int(self.grid_pos.x - 1)] == 1:
            if -self.direction != vec(-1, 0):
                poss_dir.append(vec(-1, 0))

        if GRID[int(self.grid_pos.y)][int(self.grid_pos.x + 1)] == 1:
            if -self.direction != vec(1, 0):
                poss_dir.append(vec(1, 0))

        # choose a direction randomly from available changes
        if len(poss_dir) != 0:
            rand = random.randint(0, len(poss_dir) - 1)
            self.direction = poss_dir[rand]

    def calcGridPos(self):
        if self.direction.x != 0:
            # Check if the ghost has entered a new grid tile and update direction as necessary
            if (self.pixel_pos.x - 30) % CELL_W == 0:
                self.grid_pos.x = (self.pixel_pos.x - CELL_W // 2) // CELL_W

        if self.direction.y != 0:
            # Check if the ghost has entered a new grid tile and update direction as necessary
            if (self.pixel_pos.y - 55) % CELL_H == 0:
                self.grid_pos.y = (self.pixel_pos.y - (CELL_H // 2) - PAD_TOP) // CELL_H

    def step(self):
        if self.steps == 0:
            self.check_tile()
            self.steps = 20

        if self.steps > 0:
            self.pixel_pos += self.direction
            self.image_pos += self.direction
            self.calcGridPos()
            self.steps -= 1

    def update(self):
        # Update the animation frame every 15 game frames
        self.frame_count += 1

        if self.frame_count % 15 == 0:
            self.update_frame()

        if self.frame_count % 2 == 0:
            if self.should_display:
                if self.ghost_alive:
                    if not self.power_pellet_active:
                        if self.aggressive:
                            if self.steps == 0:
                                if len(self.path) > 0:
                                    if self.path[-1] != self.pac_pos:
                                        self.path = self.a_search(self.grid_pos, self.pac_pos)
                                    if len(self.path) > 0:
                                        if self.grid_pos != self.path[0]:
                                            self.direction = vec(0, 0)
                                            self.direction = self.path[0] - self.grid_pos
                                else:
                                    self.path = self.a_search(self.grid_pos, self.pac_pos)
                                # Before resetting steps, check that the direction isn't pointed into a wall
                                if not self.check_wall_collision():
                                    self.steps = 20
                            if self.steps > 0:
                                self.pixel_pos += self.direction
                                self.image_pos += self.direction
                                self.calcGridPos()
                                if len(self.path) > 0:
                                    if self.grid_pos == self.path[0]:
                                        self.path.pop(0)
                                self.steps -= 1
                        else:
                            self.step()
                    else:
                        self.step()

                else:
                    # Movement when the ghost is dead
                    if not self.respawning:
                        self.path = self.a_search(self.grid_pos, vec(14, 14))
                        if self.steps == 0:
                            if len(self.path) > 0:
                                if self.grid_pos != self.path[0]:
                                    self.direction = self.path[0] - self.grid_pos
                            else:
                                self.respawning = True
                            self.steps = 20
                        if self.steps > 0:
                            self.pixel_pos += self.direction
                            self.image_pos += self.direction
                            self.calcGridPos()
                            if len(self.path) > 0:
                                if self.grid_pos == self.path[0]:
                                    self.path.pop(0)
                            self.steps -= 1
                    else:
                        self.path = self.a_search(self.grid_pos, vec(15, 11))
                        if self.steps == 0:
                            if len(self.path) > 0:
                                if self.grid_pos != self.path[0]:
                                    self.direction = self.path[0] - self.grid_pos
                            else:
                                self.respawning = False
                                self.ghost_alive = True
                            self.steps = 20
                        if self.steps > 0:
                            self.pixel_pos += self.direction
                            self.image_pos += self.direction
                            self.calcGridPos()
                            if len(self.path) > 0:
                                if self.grid_pos == self.path[0]:
                                    self.path.pop(0)
                            self.steps -= 1

    def draw(self):
        # Draw the ghost to the pygame screen
        if self.should_display:
            self.screen.blit(self.frame, self.image_pos)

    def reset(self, index):
        self.direction = vec(0, 0)

        # todo: set them to their original spawns
        if index == 0:
            self.grid_pos = vec(12, 11)
        elif index == 1:
            self.grid_pos = vec(15, 11)
        elif index == 2:
            self.grid_pos = vec(12, 17)
        elif index == 3:
            self.grid_pos = vec(15, 17)


        # pixel position of center of ghost
        self.pixel_pos = vec(self.grid_pos.x * CELL_W + (CELL_W // 2),
                             self.grid_pos.y * CELL_H + (CELL_H // 2) + PAD_TOP)
        # pixel position for top left corner of image surface
        self.image_pos = vec(self.grid_pos.x * CELL_W,
                             self.grid_pos.y * CELL_H + PAD_TOP)

        if self.aggressive:
            self.path = self.a_search(self.grid_pos, self.pac_pos)

        # Reset ghost state
        self.ghost_alive = True
        self.should_display = True

        # Reset game state
        self.power_pellet_active = False
        self.fleeing = False
        self.respawning = False

    def check_collision(self, player_bounds, pos):
        bounds = self.get_bounds();
        # Check Manhattan Distance to determine if close enough to check collisions
        # Arbitrary distance of 2 or less
        if abs(pos.x - self.grid_pos.x) + abs(pos.y - self.grid_pos.y) <= 2 and self.ghost_alive:
            # Check Players Left and Right Bounds with Ghosts Bounds
            for i in range(0, 2):
                if bounds[0].x <= player_bounds[i].x <= bounds[1].x:
                    if bounds[2].y <= player_bounds[i].y <= bounds[3].y:
                        if self.power_pellet_active:
                            self.set_alive_status(False)
                        else:
                            return True
                    else:
                        return False
            # Check Players Top and Bottom Bounds with Ghosts Bounds
            for i in range(2, 4):
                if bounds[2].y <= player_bounds[i].y <= bounds[3].y:
                    if bounds[0].x <= player_bounds[i].x <= bounds[1].x:
                        if self.power_pellet_active:
                            self.set_alive_status(False)
                        else:
                            return True
                    else:
                        return False
                return False

    def check_wall_collision(self):
        # Check that the current won't enter a wall
        if GRID[int(self.grid_pos.y + self.direction.y)][int(self.grid_pos.x + self.direction.x)] == 3:
            return True
        else:
            return False
    # -- -- -- GETTERS / SETTERS -- -- -- #
    # Currently increasing the speed creates the potential to overshoot collision detection and will need to be
    # tweaked if we wish to implement it.
    def set_speed(self, new_speed):
        self.speed = new_speed

    def get_pixel_pos(self):
        return self.pixel_pos

    def get_grid_pos(self):
        return self.grid_pos

    def set_alive_status(self, status):
        self.ghost_alive = status

    def set_display_status(self, status):
        self.should_display = status

    def set_pacman_pos(self, pos):
        self.pac_pos = pos

    def set_power_pellet_status(self, status):
        self.power_pellet_active = status

    def get_bounds(self):
        # Returns a tuple of the ghosts boundaries
        # (left, right, up, down)
        # (vec(x,y), vec(x,y), vec(x,y), vec(x,y))
        radius = CELL_W / 2
        return (vec(self.pixel_pos.x - 5, self.pixel_pos.y), vec(self.pixel_pos.x + 5, self.pixel_pos.y), \
                vec(self.pixel_pos.x, self.pixel_pos.y - radius), vec(self.pixel_pos.x, self.pixel_pos.y + radius))

    # -- -- -- A* SEARCH -- -- -- #
    # Adapted from a tutorial by A Name Not Yet Taken AB (annytab.com) for use with our structure
    # The code was published on annytab.com by 'Administrator' on January 22, 2020
    # Accessed by our team on February 27, 2021
    # Includes __eq__ and __lt__ of the class Node

    def a_search(self, start, end):
        start_node = Node(None, start)
        start_node.g = start_node.h = start_node.f = 0
        end_node = Node(None, end)
        end_node.g = end_node.h = end_node.f = 0

        moves = [vec(1, 0),
                 vec(-1, 0),
                 vec(0, 1),
                 vec(0, -1)]

        open = []
        closed = []
        open.append(start_node)

        while len(open) > 0:
            open.sort()
            current_node = open.pop(0)
            closed.append(current_node)

            if current_node == end_node:
                path = []
                while current_node != start_node:
                    path.append(current_node.position)
                    current_node = current_node.parent
                return path[::-1]

            neighbors = []
            for i in range(0, len(moves)):
                if GRID[int(current_node.position.y + moves[i].y)][int(current_node.position.x + moves[i].x)] == 1 \
                        or GRID[int(current_node.position.y + moves[i].y)][int(current_node.position.x + moves[i].x)] == 4:
                    neighbors.append(current_node.position + moves[i])

            for tile in neighbors:
                neighbor = Node(current_node, tile)
                if neighbor in closed:
                    continue

                neighbor.g = self.node_score(neighbor.position, start_node.position)
                neighbor.h = self.node_score(neighbor.position, end_node.position)
                neighbor.f = neighbor.g + neighbor.h

                if self.add_to_open(open, neighbor) == True:
                    open.append(neighbor)
        return []

    def add_to_open(self, open, neighbor):
        for node in open:
            if neighbor == node and neighbor.f >= node.f:
                return False
        return True

    def node_score(self, current, other):
        # For G score other should be the start node position
        # For H score other should be the end node position
        return abs(current.x - other.x) + abs(current.y - other.y)

class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __lt__(self, other):
        return self.f < other.f
