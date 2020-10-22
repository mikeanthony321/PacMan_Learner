import pygame
import sys
from settings import *

pygame.init()
vec = pygame.math.Vector2

class Pacman:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = 'title'
        self.load()
        
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
            self.clock.tick(FPS)
        pygame.quit()
        sys.exit()

# -- -- -- GENERAL FUNCTIONS -- -- -- #
    def load(self):
        self.level = pygame.image.load('level.png')
        self.level = pygame.transform.scale(self.level, (WIDTH, HEIGHT))

    def write(self, to_write, screen, pos, size, color, font_name):
        font = pygame.font.Font(font_name, size)
        text = font.render(to_write, False, color)
        screen.blit(text, pos)

    def write_center(self, to_write, screen, pos, size, color, font_name):
        font = pygame.font.Font(font_name, size)
        text = font.render(to_write, False, color)
        text_size = text.get_size()
        pos[0] = pos[0] - text_size[0]//2
        pos[1] = pos[1] - text_size[1] // 2
        screen.blit(text, pos)

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
        self.write('HIGH SCORE', self.screen, [115, 15], TITLE_TEXT_SIZE, WHITE, TITLE_FONT)
        self.write_center('PUSH SPACE TO START', self.screen, [WIDTH//2, HEIGHT//2], TITLE_TEXT_SIZE, GOLD, TITLE_FONT)
        self.write_center('1 PLAYER ONLY', self.screen, [WIDTH // 2, HEIGHT // 2 + 50], TITLE_TEXT_SIZE, CERU, TITLE_FONT)
        pygame.display.update()

# -- -- -- GAME FUNCTIONS -- -- -- #
    def game_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
    def game_update(self):
        pass
    def game_draw(self):
        self.screen.blit(self.level, (0,0))
        pygame.display.update()
