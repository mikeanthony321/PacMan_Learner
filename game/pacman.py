import pygame
import sys
from settings import *

pygame.init()
vec = pygame.math.Vector2

class Pacman:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.level = pygame.image.load('lev_og.png')
        self.clock = pygame.time.Clock()
        self.running = True
        self.state = 'title'
        self.analytics = Analytics()
        self.button = Button(480, 2, 70, 20)
        
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

    def grid(self):
        for x in range(GRID_W):
            pygame.draw.line(self.level, GOLD, (x*CELL_W, 0), (x*CELL_W, GRID_PIXEL_H))
        for y in range(GRID_H):
            pygame.draw.line(self.level, GOLD, (0, y*CELL_H), (WIDTH, y*CELL_H))

# -- -- -- TITLE FUNCTIONS -- -- -- #
    def title_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.button.bounds(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1])
                self.analytics.textfield.text_event(event)
            if event.type == pygame.KEYDOWN:
                self.analytics.textfield.text_event(event)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.state = 'game'
                
    def title_update(self):
        pass
    
    def title_draw(self):
        self.screen.fill(BLACK)
        self.write('HIGH SCORE', self.screen, [115, 15], TITLE_TEXT_SIZE, WHITE, TITLE_FONT)
        self.write_center('PUSH SPACE TO START', self.screen, [WIDTH//2, HEIGHT//2], TITLE_TEXT_SIZE, GOLD, TITLE_FONT)
        self.write_center('1 PLAYER ONLY', self.screen, [WIDTH // 2, HEIGHT // 2 + 50], TITLE_TEXT_SIZE, CERU, TITLE_FONT)

        if self.button.toggled:
            self.analytics.draw(self.screen)

        self.button.draw(self.screen)

        pygame.display.update()

# -- -- -- GAME FUNCTIONS -- -- -- #
    def game_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.button.bounds(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1])
    def game_update(self):
        pass
    def game_draw(self):
        self.screen.fill(BLACK)
        self.write('HIGH SCORE', self.screen, [5, 5], 13, WHITE, TITLE_FONT)
        self.screen.blit(self.level, (0, PAD_TOP))
        self.grid()

        if self.button.toggled:
            self.analytics.draw(self.screen)

        self.button.draw(self.screen)

        pygame.display.update()

class Analytics:
    def __init__(self):
        self.width = 300
        self.height = 300
        self.color = (0, 0, 0, 125)
        self.textfield = TextField(200, 200, 100, 30)
        self.frame = pygame.Surface((400,400), pygame.SRCALPHA)

    def draw(self, screen):
        self.frame.set_alpha(75)
        self.frame.fill(WHITE)
        pygame.draw.rect(self.frame, WHITE, self.frame.get_rect(), 10)

        screen.blit(self.frame, (150, 30))
        self.textfield.draw(screen)

class TextField:
    def __init__(self, x_pos, y_pos, width, height):
        self.position = (x_pos, y_pos)
        self.size = (width, height)
        self.font = pygame.font.Font(TITLE_FONT, 16)
        self.text = ''
        self.visible = False
        self.typing = False
        self.rect = pygame.Rect(self.position[0], self.position[1], self.size[0], self.size[1])
        self.text_surface = self.font.render(self.text, False, WHITE)

    def text_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.bounds(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1])
        if event.type == pygame.KEYDOWN and self.typing:
            if self.typing:
                self.text += event.unicode
                print(self.text)



    def draw(self, screen):
        pygame.draw.rect(screen, BLACK, self.rect, 1)
        self.text_surface = self.font.render(self.text, False, WHITE)
        screen.blit(self.text_surface, self.position)


    def bounds(self, cursor_x, cursor_y):
        if cursor_x > self.position[0] and cursor_y < self.position[1] + self.size[1]:
            if cursor_x < self.position[0] + self.size[0] and cursor_y > self.position[1]:
                self.typing = not self.typing
            else:
                self.typing = False

        else:
            self.typing = False

class Button:
    def __init__(self, x_pos, y_pos, width, height):
        self.position = (x_pos, y_pos)
        self.size = (width, height)
        self.toggled = False
        self.button = pygame.Surface(self.size, pygame.SRCALPHA)

    def draw(self, screen):
        self.button.set_alpha(50)
        self.button.fill(WHITE)
        pygame.draw.rect(self.button, WHITE, self.button.get_rect(), 10)
        screen.blit(self.button, self.position)

    def bounds(self, cursor_x, cursor_y):
        if cursor_x > self.position[0] and cursor_y < self.position[1] + self.size[1]:
            if cursor_x < self.position[0] + self.size[0] and cursor_y > self.position[1]:
                self.toggled = not self.toggled

