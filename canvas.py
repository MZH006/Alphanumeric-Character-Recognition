import pygame
import numpy as np

WIDTH, HEIGHT = 280, 280
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def run_canvas():
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("Arial", 24)
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Draw Character")
    clock = pygame.time.Clock()
    drawing = False
    screen.fill(BLACK)
    last_pos = None


    def draw_brush(current_pos):
        nonlocal last_pos
        if last_pos is not None:
            pygame.draw.line(screen, WHITE, last_pos, current_pos, 10)
        last_pos = current_pos


    running = True
    done_drawing = False

    while running and not done_drawing:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True

            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                last_pos = None
                done_drawing = True 

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    screen.fill(BLACK)

        if drawing:
            x, y = pygame.mouse.get_pos()
            draw_brush((x, y))

        pygame.display.flip()
        clock.tick(60)


    return screen 

    
