import pygame
import sys

# Initialize Pygame
pygame.init()

# Window dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Rectangle Outline Example")

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Rectangle parameters
rect_x = 100
rect_y = 100
rect_width = 200
rect_height = 150
rect_border_width = 5  # Thickness of the border

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the background
    screen.fill(WHITE)

    # Draw the rectangle outline
    pygame.draw.rect(
        screen,
        RED,  # Border color
        (rect_x, rect_y, rect_width, rect_height),  # Rectangle dimensions
        rect_border_width  # Border thickness
    )

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
