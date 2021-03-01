import sys
from api.game_agent import GameAgentAPI
from player import *
from cell import *
from analytics_frame_2 import *
from network_visualizer_test import get_network_diagram

pygame.init()
vec = pygame.math.Vector2

class Pacman(GameAgentAPI):
    def __init__(self, monitor_size):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.analytics = Analytics(monitor_size, get_network_diagram())
        self.level = pygame.image.load('lev_og.png')
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = 'title'
        self.cells = CellMap()
        self.player = Player(self, PLAYER_START_POS)
        
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

            #This is a temporary section meant to test interaction (albeit a simple interaction)
            #self.analytics.updateScreen()
            if self.analytics.running:
                self.state = 'game'

            self.clock.tick(FPS)

        # exit routine
        if self.player.score >= int(HIGH_SCORE):
            open("db/hs.txt", "w").write(str(self.player.score))
        if self.player.score >= int(self.analytics.tar_high_score):
            print("Target High Score Achieved!")
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
        self.player.reset()

    def spawn_coins(self):
        for cell in self.cells.map:
            if cell.hasCoin:
                if cell.coin.isSuperCoin:
                    pygame.draw.circle(self.screen, RED, (
                        cell.pos[0] * CELL_W + CELL_W // 2, cell.pos[1] * CELL_H + CELL_H // 2 + PAD_TOP), 6)
                else:
                    pygame.draw.circle(self.screen, WHITE,
                                       (cell.pos[0] * CELL_W + CELL_W // 2,
                                        cell.pos[1] * CELL_H + CELL_H // 2 + PAD_TOP), 3)

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

    def game_update(self):
        self.player.update()

    def game_draw(self):
        self.screen.fill(BLACK)

        # top bar
        self.write("HIGH " + HIGH_SCORE, self.screen, [5, 5], 13, WHITE, TITLE_FONT)
        self.write("SCORE " + str(self.player.score), self.screen, [200, 5], 13, WHITE, TITLE_FONT)
        self.write("DEATHS " + str(self.player.deaths), self.screen, [395, 5], 13, WHITE, TITLE_FONT)

        # level
        self.screen.blit(self.level, (0, PAD_TOP))
        self.spawn_coins()

        # spawn
        self.player.draw()
        if SHOW_GRID:
            self.grid()
        pygame.display.update()

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
