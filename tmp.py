import pygame
import sys

# Initialize Pygame
pygame.init()

# Set up the screen
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption('Pygame Button with Hover and Click Effects')

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 200, 0)
LIGHT_GREEN = (144, 238, 144)

# Define the Button class
class Button:
    def __init__(self, x, y, width, height, text, color):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.default_color = color
        self.hover_color = LIGHT_GREEN
        self.click_color = DARK_GREEN
        self.current_color = color
        self.font = pygame.font.Font(None, 36)
        self.clicked = False
    
    def draw(self, screen):
        # Draw the button
        pygame.draw.rect(screen, self.current_color, (self.x, self.y, self.width, self.height), 2)
        
        # Render the text
        text_surface = self.font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        screen.blit(text_surface, text_rect)
    
    def is_hovered(self, mouse_pos):
        # Check if the button is hovered over
        return self.x <= mouse_pos[0] <= self.x + self.width and self.y <= mouse_pos[1] <= self.y + self.height

    def handle_event(self, event):
        mouse_pos = pygame.mouse.get_pos()
        
        # Check if the button is hovered
        if self.is_hovered(mouse_pos):
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.current_color = self.click_color
                self.clicked = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if self.clicked:
                    self.current_color = self.hover_color
                    self.clicked = False
                    print("Button Clicked!")
            else:
                self.current_color = self.hover_color
        else:
            self.current_color = self.default_color

# Create a button instance
button = Button(300, 250, 200, 80, "Click Me", GREEN)

# Main game loop
running = True
while running:
    screen.fill(WHITE)  # Fill the screen with a white background

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        button.handle_event(event)

    # Draw the button
    button.draw(screen)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
