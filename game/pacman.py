import sys
import copy
import random
import numpy as np
from api.game_agent import GameAgentAPI
from agent import LearnerAgent
from player import *
from cell import *
from ghost import *
from settings import *
from api.actions import Actions

from analytics_frame import *

pygame.init()
vec = pygame.math.Vector2

class Pacman(GameAgentAPI):
    def __init__(self, monitor_size):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.level = pygame.image.load('res/lev_og.png')
        self.sprites = pygame.image.load('res/pacmanspritesheet.png')
        self.clock = pygame.time.Clock()
        self.running = True
        self.app_state = 'title'
        self.tar_high_score = None

        self.cells = CellMap()

        self.player_start = PLAYER_START_POS
        self.player_respawn = PLAYER_RESPAWN_POS
        self.blinky_start = BLINKY_START_POS
        self.inky_start = INKY_START_POS
        self.pinky_start = PINKY_START_POS
        self.clyde_start = CLYDE_START_POS
        self.tar_high_score = 0
        self.ghosts = []
        self.centered_start_pos = False

        self.ghosts.append(Ghost(self, self.screen, True, "Blinky", vec(12, 11), BLINKY_SPRITE_POS, self.sprites))
        self.ghosts.append(Ghost(self, self.screen, False, "Inky", vec(15, 11), INKY_SPRITE_POS, self.sprites))
        self.ghosts.append(Ghost(self, self.screen, False, "Pinky", vec(12, 17), PINKY_SPRITE_POS, self.sprites))
        self.ghosts.append(Ghost(self, self.screen, False, "Clyde", vec(15, 17), CLYDE_SPRITE_POS, self.sprites))

        self.power_pellet_timer = POWER_PELLET_TIMER
        self.idle_timer = 0
        self.set_game_objects()

    def set_game_objects(self):
        self.player = Player(self, self.screen, self.player_start, self.player_respawn, self.sprites)
        self.ghosts = []

        self.ghosts.append(Ghost(self, self.screen, True, "Blinky", self.blinky_start, BLINKY_SPRITE_POS, self.sprites))
        self.ghosts.append(Ghost(self, self.screen, False, "Inky", self.inky_start, INKY_SPRITE_POS, self.sprites))
        self.ghosts.append(Ghost(self, self.screen, False, "Pinky", self.pinky_start, PINKY_SPRITE_POS, self.sprites))
        self.ghosts.append(Ghost(self, self.screen, False, "Clyde", self.clyde_start, CLYDE_SPRITE_POS, self.sprites))

        # self.move_history = []
        # for i in range(IDLE_HISTORY_LENGTH):
        #     self.move_history.append(None)

    def run(self):
        while self.running:
            if self.app_state == 'title':
                self.title_events()
                self.title_update()
                self.title_draw()
            elif self.app_state == 'game':
                self.game_events()
                self.game_update()
                self.game_draw()
            else:
                self.running = False

            self.clock.tick(FPS)

        # exit routine
        self.score_reset()
        pygame.quit()
        sys.exit()

# -- -- -- GENERAL FUNCTIONS -- -- -- #
    def write(self, to_write, screen, pos, size, color, font_name):
        font = pygame.font.Font(font_name, size)
        text = font.render(to_write, False, color)
        screen.blit(text, pos)

    def write_center(self, to_write, screen, pos, size, color, font_name):
        font = pygame.font.Font(font_name, size)
        text = font.render(to_write, False, color)
        text_size = text.get_size()
        pos[0] = pos[0] - text_size[0] // 2
        pos[1] = pos[1] - text_size[1] // 2
        screen.blit(text, pos)

    def grid(self):
        for x in range(GRID_W):
            pygame.draw.line(self.level, GOLD, (x * CELL_W, 0), (x * CELL_W, GRID_PIXEL_H))
        for y in range(GRID_H):
            pygame.draw.line(self.level, GOLD, (0, y * CELL_H), (WIDTH, y * CELL_H))

    def reset_level(self):
        self.cells = CellMap()
        self.score_reset()
        self.player.reset(self.centered_start_pos)
        self.player.set_alive_status(True)
        self.player.set_game_over_status(False)
        self.ghost_reset()

    def score_reset(self):
        if self.player.score >= int(HIGH_SCORE):
            open("db/hs.txt", "w").write(str(self.player.score))

        if self.player.score >= int(self.tar_high_score):
            print("Target High Score Achieved! Score: ", self.player.score)
            Analytics.analytics_instance.help_text_label.setText("Target High Score Achieved! Score: " + str(self.player.score))
        self.player.score = 0

    def spawn_coins(self):
        count = 0
        for cell in self.cells.map:
            if cell.hasCoin:
                count += 1
                if cell.coin.isSuperCoin:
                    pygame.draw.circle(self.screen, RED, (
                        cell.pos[0] * CELL_W + CELL_W // 2, cell.pos[1] * CELL_H + CELL_H // 2 + PAD_TOP), 6)
                else:
                    pygame.draw.circle(self.screen, WHITE,
                                       (cell.pos[0] * CELL_W + CELL_W // 2,
                                        cell.pos[1] * CELL_H + CELL_H // 2 + PAD_TOP), 3)
        if count == 0:
            self.reset_level()

# -- -- -- TITLE FUNCTIONS -- -- -- #
    def title_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.app_state = 'game'
                
    def title_update(self):
        pass
    
    def title_draw(self):
        self.screen.fill(BLACK)
        self.write("HIGH SCORE " + HIGH_SCORE, self.screen, [115, 15], TITLE_TEXT_SIZE, WHITE, TITLE_FONT)
        self.write_center('PUSH SPACE TO START', self.screen, [WIDTH//2, HEIGHT//2], TITLE_TEXT_SIZE, GOLD, TITLE_FONT)
        self.write_center('1 PLAYER ONLY', self.screen, [WIDTH // 2, HEIGHT // 2 + 50], TITLE_TEXT_SIZE, BLUE, TITLE_FONT)
        pygame.display.update()

# -- -- -- GAME FUNCTIONS -- -- -- #

    def game_update(self):
        # Check whether the stop button has been pressed
        if not Analytics.get_running_state():
            LearnerAgent.agent_instance.reset_agent()
            self.stop_simulation()
        # Check whether the target high score has been reached
        if self.player.score >= self.tar_high_score:
            Analytics.non_user_stop()
            self.stop_simulation()

        self.set_pac_pos()
        self.check_ghost_pac_collision()

        if not self.player.get_game_over_status():
            player_pos = copy.deepcopy(self.player.get_grid_pos())
            self.player.update()

            # Alert the AI if a new grid square has been entered
            if (self.player.get_grid_pos() != player_pos) or self.idle_timer >= MAX_IDLE_ALLOWANCE:
                self.idle_timer = 0

                # for i in range(len(self.move_history) - 1):
                #     self.move_history[i] = copy.deepcopy(self.move_history[i + 1])
                # self.move_history[len(self.move_history) - 1] = self.player.get_grid_pos()

                if random.random() < DECISION_FREQUENCY:
                    LearnerAgent.run_decision()
                    Analytics.update_frame()
            else:
                self.idle_timer += 1

            if not self.player.alive:
                self.reset_level()

        # When Pacman hits a Super Coin, the player pow pel status
        # flips to true and back to false upon collecting the next coin.
        # This is managed during coin collection in player.py
            self.player.power_pellet_timer()

            if self.player.power_pellet_active:
                self.set_ghost_power_pellet_status(True)
            else:
                self.set_ghost_power_pellet_status(False)

            if self.player.get_alive_status():
                for i in range(len(self.ghosts)):
                    self.ghosts[i].update()

    def game_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False


    def game_draw(self):
        self.screen.fill(BLACK)

        # top bar
        self.write("HIGH " + HIGH_SCORE, self.screen, [5, 5], 13, WHITE, TITLE_FONT)
        self.write("SCORE " + str(self.player.score), self.screen, [200, 5], 13, WHITE, TITLE_FONT)
        self.write("DEATHS " + str(self.player.deaths), self.screen, [395, 5], 13, WHITE, TITLE_FONT)

        # level
        self.screen.blit(self.level, (0, PAD_TOP))
        self.spawn_coins()

        # This if/else alters the order of drawing such that a ghost will appear over Pac-Man normally,
        # but render below when Pac-Man is defeated. Might be a waste to do this, we can just render Pac-Man last at
        # all times if preferred.
        if self.player.alive:
            # spawn
            self.player.draw()

            # ghosts
            for i in range(len(self.ghosts)):
                self.ghosts[i].draw()

        else:
            # ghosts
            for i in range(len(self.ghosts)):
                self.ghosts[i].draw()

            # spawn
            self.player.draw()

        if SHOW_GRID:
            self.grid()

        pygame.display.update()

    def set_pac_pos(self):
        for i in range(len(self.ghosts)):
            self.ghosts[i].set_pacman_pos(self.player.get_grid_pos())

    def check_ghost_pac_collision(self):
        for i in range(len(self.ghosts)):
            if self.player.get_alive_status():
                # Returns true when a collision occurs and a Power Pellet is not active
                if self.ghosts[i].check_collision(self.player.get_bounds(), self.player.get_grid_pos()):
                    self.player.set_alive_status(False)

    def set_ghost_power_pellet_status(self, status):
        for i in range(len(self.ghosts)):
            self.ghosts[i].set_power_pellet_status(status)
            
    def ghost_reset(self):
        for i in range(len(self.ghosts)):
            self.ghosts[i].reset(i)

    def stop_simulation(self):
        # Does not currently set Pac-man to original spawn, just randomizes so will need to account
        # for this once we've decided what spawning options we want
        self.reset_level()
        self.app_state = "title"

# -- -- -- AGENT API FUNCTIONS -- -- -- #

    def gameStart(self):
        self.app_state = 'game'

    def setTarHighScore(self, score):
        self.tar_high_score = score

    def set_start_pos(self, pos_dict, centered_start):
        self.player_start = pos_dict['player_start']
        self.player_respawn= pos_dict['player_respawn']
        self.blinky_start = pos_dict['blinky']
        self.inky_start = pos_dict['inky']
        self.pinky_start = pos_dict['pinky']
        self.clyde_start = pos_dict['clyde']
        self.centered_start_pos = centered_start
        self.set_game_objects()

    def getAvailableActions(self, prev_decision):
        available_actions = []
        player_x, player_y = self.player.get_grid_pos()
        for action, x, y, opposite in [(Actions.UP, 0, -1, Actions.DOWN), (Actions.DOWN, 0, 1, Actions.UP),
                                       (Actions.LEFT, -1, 0, Actions.RIGHT), (Actions.RIGHT, 1, 0, Actions.LEFT)]:
            cell = next((c for c in self.cells.map if c.pos == (player_x + x, player_y + y)), None)
            if (cell is not None) and (not cell.hasWall) and (prev_decision is not opposite):
                available_actions.append(action)

        return available_actions

    def moveUp(self):
        self.player.move(vec(0, -1))

    def moveDown(self):
        self.player.move(vec(0, 1))

    def moveLeft(self):
        self.player.move(vec(-1, 0))

    def moveRight(self):
        self.player.move(vec(1, 0))

    def getGhostsGridCoords(self):
        player_coords = self.player.get_grid_pos()
        ghost_coords = []
        for ghost in self.ghosts:
            coords = ghost.get_grid_pos()
            ghost_coords.append(coords - player_coords)
        return ghost_coords

    def getNearestPelletGridCoords(self):
        player_coords = self.player.get_grid_pos()
        min_coords = [100, 100]
        for cell in [c for c in self.cells.map if c.hasCoin]:
            cell_coords = cell.pos
            distances = [sum([abs(c) for c in min_coords - player_coords]),
                         sum([abs(c) for c in cell_coords - player_coords])]
            if distances[1] < distances[0]:
                min_coords = cell_coords
        return min_coords - player_coords

    def getNearestPowerPelletGridCoords(self):
        player_coords = self.player.get_grid_pos()
        min_coords = [100, 100]
        for cell in [c for c in self.cells.map if (c.hasCoin and c.coin.isSuperCoin)]:
            cell_coords = cell.pos
            distances = [sum([abs(c) for c in min_coords - player_coords]),
                         sum([abs(c) for c in cell_coords - player_coords])]
            if distances[1] < distances[0]:
                min_coords = cell_coords
        return min_coords - player_coords

    def isPowerPelletActive(self):
        return self.player.power_pellet_active

    def getReward(self):
        reward = 0
        player_cell = next((c for c in self.cells.map if c.pos == self.player.get_grid_pos()), None)
        player_in_coin_cell = player_cell is not None and player_cell.hasCoin
        pellet_count = len([c for c in self.cells.map if not c.hasWall and not c.hasCoin])

        reward += Q_PELLET_PROXIMITY_FACTOR / (np.linalg.norm(np.array(self.getNearestPelletGridCoords()) - np.array([0, 0])) + 1)
        # for i in range(1, len(self.move_history)):
        #     if (self.move_history[i] is not None) and (self.move_history[i] == self.move_history[0]):
        #         reward -= Q_IDLE_PENALTY
        if player_in_coin_cell:
            reward += Q_PELLET_FUNC(pellet_count)
            if len([c for c in self.cells.map if c.hasCoin]) == 1:
                reward += Q_LEVEL_PASSED
        for ghost in self.ghosts:
            distance = np.linalg.norm(np.array(ghost.get_grid_pos()) - np.array(self.player.get_grid_pos()))
            reward += (Q_GHOST_PROXIMITY_FACTOR / (distance + 1)) * (1 if ghost.power_pellet_active else -1)

        #print(reward)
        return reward
