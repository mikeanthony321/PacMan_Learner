import sys
from player import *
from cell import *
from analytics import *
from ghost import *

pygame.init()
vec = pygame.math.Vector2


class Pacman:
    def __init__(self, monitor_size):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.analytics = Analytics(monitor_size)
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
        if self.player.score >= int(HIGH_SCORE):
            open("db/hs.txt", "w").write(str(self.player.score))
        # todo: this ^ is just a mock high score counter, it adds 1 to the high score in db on each exit

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
        self.write_center('PUSH SPACE TO START', self.screen, [WIDTH // 2, HEIGHT // 2], TITLE_TEXT_SIZE, GOLD,
                          TITLE_FONT)
        self.write_center('1 PLAYER ONLY', self.screen, [WIDTH // 2, HEIGHT // 2 + 50], TITLE_TEXT_SIZE, CERU,
                          TITLE_FONT)
        pygame.display.update()

    # -- -- -- GAME FUNCTIONS -- -- -- #
    def game_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.player.move(vec(-1, 0))
                if event.key == pygame.K_RIGHT:
                    self.player.move(vec(1, 0))
                if event.key == pygame.K_UP:
                    self.player.move(vec(0, -1))
                if event.key == pygame.K_DOWN:
                    self.player.move(vec(0, 1))

    def game_update(self):
        self.player.update()

        self.blinky.update()
        self.pinky.update()
        self.inky.update()
        self.clyde.update()

        self.checkGhostPacCollision()

    def game_draw(self):
        self.screen.fill(BLACK)

        # top bar
        self.write("HIGH " + HIGH_SCORE, self.screen, [5, 5], 13, WHITE, TITLE_FONT)
        self.write("SCORE " + str(self.player.score), self.screen, [200, 5], 13, WHITE, TITLE_FONT)
        self.write("DEATHS " + str(self.player.deaths), self.screen, [395, 5], 13, WHITE, TITLE_FONT)

        # level
        self.screen.blit(self.level, (0, PAD_TOP))
        self.spawn_coins()

        # This if/else renders alters the order of drawing such that a ghost will appear over Pac-Man normally,
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

    def checkGhostPacCollision(self):
        # todo: this could be much better
        if self.blinky.getPixelPos() == self.player.getPixelPos() and self.player.getAliveStatus():
            self.player.setAliveStatus(False)
            self.blinky.setDisplayStatus(False)
        if self.pinky.getPixelPos() == self.player.getPixelPos() and self.player.getAliveStatus():
            self.player.setAliveStatus(False)
            self.pinky.setDisplayStatus(False)
        if self.inky.getPixelPos() == self.player.getPixelPos() and self.player.getAliveStatus():
            self.player.setAliveStatus(False)
            self.inky.setDisplayStatus(False)
        if self.clyde.getPixelPos() == self.player.getPixelPos() and self.player.getAliveStatus():
            self.player.setAliveStatus(False)
            self.clyde.setDisplayStatus(False)

        if not self.player.getAliveStatus():
            self.blinky.setActiveStatus(False)
            self.pinky.setActiveStatus(False)
            self.inky.setActiveStatus(False)
            self.clyde.setActiveStatus(False)
