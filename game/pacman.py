import sys
from api.game_agent import GameAgentAPI
from player import *
from cell import *
from ghost import *
from analytics_frame_2 import *
from network_visualizer_test import get_network_diagram

pygame.init()
vec = pygame.math.Vector2

class Pacman(GameAgentAPI):
    def __init__(self, monitor_size):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.analytics = Analytics(monitor_size, get_network_diagram())
        self.level = pygame.image.load('res/lev_og.png')
        self.sprites = pygame.image.load('res/pacmanspritesheet.png')
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = 'title'
        self.cells = CellMap()

        self.player = Player(self, self.screen, PLAYER_START_POS, self.sprites)

        self.blinky = Ghost(self, self.screen, True, "Blinky", BLINKY_START_POS, BLINKY_SPRITE_POS, self.sprites)
        self.inky = Ghost(self, self.screen, False, "Inky", INKY_START_POS, INKY_SPRITE_POS, self.sprites)
        self.pinky = Ghost(self, self.screen, False, "Pinky", PINKY_START_POS, PINKY_SPRITE_POS, self.sprites)
        self.clyde = Ghost(self, self.screen, False, "Clyde", CLYDE_START_POS, CLYDE_SPRITE_POS, self.sprites)
        self.power_pellet_timer = POWER_PELLET_TIMER

    def run(self):
        while self.running:
            if self.state == 'title':
                self.title_events()
                self.title_update()
                self.title_draw()
            elif self.state == 'game':
                self.game_events()
                self.game_update()
                self.game_draw()
            else:
                self.running = False

            if self.analytics.running:
                self.state = 'game'

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
        self.player.reset()
        self.player.set_alive_status(True)
        self.player.set_game_over_status(False)
        self.ghost_reset()
        self.analytics.setRestart(False)

    def score_reset(self):
        if self.player.score >= int(HIGH_SCORE):
            open("db/hs.txt", "w").write(str(self.player.score))

        if self.player.score >= int(self.analytics.tar_high_score):
            print("Target High Score Achieved! Score: ", self.player.score)

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
                self.state = 'game'
                
    def title_update(self):
        pass
    
    def title_draw(self):
        self.screen.fill(BLACK)
        self.write("HIGH SCORE " + HIGH_SCORE, self.screen, [115, 15], TITLE_TEXT_SIZE, WHITE, TITLE_FONT)
        self.write_center('PUSH SPACE TO START', self.screen, [WIDTH//2, HEIGHT//2], TITLE_TEXT_SIZE, GOLD, TITLE_FONT)
        self.write_center('1 PLAYER ONLY', self.screen, [WIDTH // 2, HEIGHT // 2 + 50], TITLE_TEXT_SIZE, BLUE, TITLE_FONT)
        pygame.display.update()

# -- -- -- GAME FUNCTIONS -- -- -- #
    def game_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.moveLeft()
                if event.key == pygame.K_RIGHT:
                    self.moveRight()
                if event.key == pygame.K_UP:
                    self.moveUp()
                if event.key == pygame.K_DOWN:
                    self.moveDown()
                if self.player.get_game_over_status == True and event.key == pygame.K_SPACE:
                    self.reset_level()

    def game_update(self):
        if not self.player.get_game_over_status():
            self.player.update()
            #if not self.player.alive:
                #self.reset_level()

        # When Pacman hits a Super Coin, the player pow pel status
        # flips to true and back to false upon collecting the next coin.
        # This is managed during coin collection in player.py

            if self.player.power_pellet_active:
                if self.power_pellet_timer == POWER_PELLET_TIMER:
                    self.set_ghost_power_pellet_status(True)
                if self.power_pellet_timer > 0:
                    self.power_pellet_timer -= 1
                else:
                    self.player.set_power_pellet_status(False)
                    self.set_ghost_power_pellet_status(False)
            else:
                self.power_pellet_timer = POWER_PELLET_TIMER

            if self.player.get_alive_status():
                self.blinky.update()
                self.pinky.update()
                self.inky.update()
                self.clyde.update()

            self.set_pac_pos()

            self.check_ghost_pac_collision()

        if self.analytics.getRestart() == True:
            self.reset_level()

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
            self.blinky.draw()
            self.pinky.draw()
            self.inky.draw()
            self.clyde.draw()
        else:
            # ghosts
            self.blinky.draw()
            self.pinky.draw()
            self.inky.draw()
            self.clyde.draw()

            # spawn
            self.player.draw()

        if SHOW_GRID:
            self.grid()

        pygame.display.update()

    def set_pac_pos(self):
        self.blinky.set_pacman_pos(self.player.get_presence())
        self.inky.set_pacman_pos(self.player.get_presence())
        self.pinky.set_pacman_pos(self.player.get_presence())
        self.clyde.set_pacman_pos(self.player.get_presence())

    def check_ghost_pac_collision(self):
        # todo: this could be much better
        if self.blinky.get_pixel_pos() == self.player.get_pixel_pos() and self.player.get_alive_status():
            if not self.player.power_pellet_active:
                self.player.set_alive_status(False)
                self.blinky.set_display_status(False)
            else:
                self.blinky.set_alive_status(False)
        if self.pinky.get_pixel_pos() == self.player.get_pixel_pos() and self.player.get_alive_status():
            if not self.player.power_pellet_active:
                self.player.set_alive_status(False)
                self.pinky.set_display_status(False)
            else:
                self.pinky.set_alive_status(False)
        if self.inky.get_pixel_pos() == self.player.get_pixel_pos() and self.player.get_alive_status():
            if not self.player.power_pellet_active:
                self.player.set_alive_status(False)
                self.inky.set_display_status(False)
            else:
                self.inky.set_alive_status(False)
        if self.clyde.get_pixel_pos() == self.player.get_pixel_pos() and self.player.get_alive_status():
            if not self.player.power_pellet_active:
                self.player.set_alive_status(False)
                self.clyde.set_display_status(False)
            else:
                self.clyde.set_alive_status(False)

        if self.player.get_alive_status() == False:
            self.analytics.setRunning(False)

    def set_ghost_power_pellet_status(self, status):
        self.blinky.set_power_pellet_status(status)
        self.inky.set_power_pellet_status(status)
        self.pinky.set_power_pellet_status(status)
        self.clyde.set_power_pellet_status(status)


    def ghost_reset(self):
        self.blinky.reset()
        self.pinky.reset()
        self.inky.reset()
        self.clyde.reset()

# -- -- -- AGENT API FUNCTIONS -- -- -- #
    def getUpdateState(self):
        # Implement me!
        pass

    def moveUp(self):
        self.player.move(vec(0, -1))

    def moveDown(self):
        self.player.move(vec(0, 1))

    def moveLeft(self):
        self.player.move(vec(-1, 0))

    def moveRight(self):
        self.player.move(vec(1, 0))

    def getPlayerGridCoords(self):
        # Implement me!
        pass

    def getNearestGhostGridCoords(self):
        # Implement me!
        pass

    def getNearestPelletGridCoords(self):
        # Implement me!
        pass

    def getNearestPowerPelletGridCoords(self):
        # Implement me!
        pass

    def isPowerPelletActive(self):
        # Implement me!
        pass
