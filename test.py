import math, random
import pygame, sys
from pygame.locals import *

colorType = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (255, 165, 0),   # Orange
    (128, 0, 128),   # Purple
    (0, 128, 128),   # Teal
    (255, 192, 203), # Pink
]

colorTypeSurface = []

for color in colorType:
    rect_surface = pygame.Surface((40,40), pygame.SRCALPHA)
    rect_surface.fill((*color,200))
    colorTypeSurface.append(rect_surface)

class AttractiveGrid:
    def __init__(self, grid=[], m=6):
        self.grid = grid
        self.m = m
        self.font = pygame.font.Font(None, 22)
        self.rect_surface = pygame.Surface((40,40), pygame.SRCALPHA)
        self.rect_surface.fill((0,0,0,200))
    
    def render(self, SCREEN):
        SCREEN.blit(self.rect_surface, (10, 10, 40, 40))
        pygame.draw.rect(SCREEN, (255,255,255), pygame.Rect(10, 10, 40, 40), 1)
        
        for i in range(1,self.m):
            SCREEN.blit(colorTypeSurface[i], (10 + 40 * i, 10, 40, 40))
        
        for j in range(1,self.m):
            SCREEN.blit(colorTypeSurface[j], (10, 10 + 40 * j, 40, 40))
        
        for i in range(1,self.m):
            for j in range(1,self.m):
                SCREEN.blit(self.rect_surface, (10 + 40 * i, 10 + 40 * j, 40, 40))
                pygame.draw.rect(SCREEN, (255,255,255), pygame.Rect(10 + 40 * i, 10 + 40 * j, 40, 40), 1)
                tmp_text = self.font.render(f"{self.grid[i][j]:.2f}", True, (255, 255, 255))
                w,h = tmp_text.get_size()
                SCREEN.blit(tmp_text, (10 + 40 * (i + 1) - w - 3, 10 + 40 * j + (44 - h) / 2))

class Simulation:
    def __init__(self, n = 1000, dt = 0.02, frictionHalfLife = 0.04, rMax = 0.1, m = 6, forceFactor = 10):
        self.n = n
        self.dt = dt
        self.frictionHalfLife = frictionHalfLife
        self.rMax = rMax
        self.m = m
        self.forceFactor = forceFactor

        self.matrix = [[random.uniform(-1, 1) for _ in range(self.m)] for _ in range(self.m)]

        self.frictionFactor = math.pow(0.5, self.dt / self.frictionHalfLife)

        self.colors = []
        self.positionsX = []
        self.positionsY = []
        self.velocitiesX = [0.0] * self.n
        self.velocitiesY = [0.0] * self.n

        for i in range(self.n):
            self.colors.append(random.randint(0, self.m - 1))
            self.positionsX.append(random.random())
            self.positionsY.append(random.random())
    
    def force(self, r, a):
        beta = 0.3
        if r < beta:
            return r / beta - 1
        elif beta < r and r < 1:
            return a * (1 - abs(2 * r - 1 - beta) / (1 - beta))
        else:
            return 0

    def updateParticles(self):
        # Grid parameters
        cell_size = self.rMax
        num_cells_x = int(1 / cell_size)
        num_cells_y = int(1 / cell_size)
        grid = {}

        # Build the grid with periodic boundary conditions
        for i in range(self.n):
            cell_x = int(self.positionsX[i] / cell_size) % num_cells_x
            cell_y = int(self.positionsY[i] / cell_size) % num_cells_y
            key = (cell_x, cell_y)
            if key not in grid:
                grid[key] = []
            grid[key].append(i)

        # Update particles
        for i in range(self.n):
            totalForceX = 0.0
            totalForceY = 0.0
            cell_x = int(self.positionsX[i] / cell_size) % num_cells_x
            cell_y = int(self.positionsY[i] / cell_size) % num_cells_y

            # Check neighboring cells with wrap-around
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    neighbor_x = (cell_x + dx) % num_cells_x
                    neighbor_y = (cell_y + dy) % num_cells_y
                    neighbor_key = (neighbor_x, neighbor_y)
                    if neighbor_key in grid:
                        for j in grid[neighbor_key]:
                            if i == j:
                                continue

                            rx = self.positionsX[j] - self.positionsX[i]
                            ry = self.positionsY[j] - self.positionsY[i]

                            # Apply periodic boundary conditions for positions
                            rx -= round(rx)
                            ry -= round(ry)

                            r = math.hypot(rx, ry)

                            if r > 0 and r < self.rMax:
                                f = self.force(r / self.rMax, self.matrix[self.colors[i]][self.colors[j]])
                                totalForceX += rx / r * f
                                totalForceY += ry / r * f

            totalForceX *= self.rMax * self.forceFactor
            totalForceY *= self.rMax * self.forceFactor

            self.velocitiesX[i] *= self.frictionFactor
            self.velocitiesY[i] *= self.frictionFactor

            self.velocitiesX[i] += totalForceX * self.dt
            self.velocitiesY[i] += totalForceY * self.dt

        for i in range(self.n):
            self.positionsX[i] += self.velocitiesX[i] * self.dt
            self.positionsY[i] += self.velocitiesY[i] * self.dt

            # Keep positions within [0,1) using modulo operator
            self.positionsX[i] %= 1
            self.positionsY[i] %= 1

    def renderScreen(self, SCREEN):
        SCREEN.fill((0, 0, 0))

        for i in range(self.n):
            screenX = int(self.positionsX[i] * WIDTH)
            screenY = int(self.positionsY[i] * HEIGHT)
            pygame.draw.circle(SCREEN, colorType[self.colors[i]], (screenX, screenY), 2)

pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 600
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Particle Life Simulation")

clock = pygame.time.Clock()

def menuScreen():
    running = True
    currSimul = Simulation(m=random.randint(2,3))
    
    titleH1 = pygame.font.Font(None, 72)
    text_titleH1 = titleH1.render("Particle Life", True, (255, 255, 255))    
    wH1,hH1 = text_titleH1.get_size()
    
    titleP = pygame.font.Font(None, 36)
    text_titleP = titleP.render("Space to Start", True, (255, 255, 255))    
    wP,hP = text_titleP.get_size()
    
    x_positions = [(WIDTH - wH1) // 2, (WIDTH - wP) // 2]
    y_positions = [(HEIGHT - hH1) // 7*3, (HEIGHT - hP) // 7*4]
    
    rect_surface = pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA)
    rect_surface.fill((0,0,0,100))
    
    while running:
        currSimul.updateParticles()
        currSimul.renderScreen(SCREEN)
        SCREEN.blit(rect_surface, (0,0,WIDTH,HEIGHT))
        
        SCREEN.blit(text_titleH1, (x_positions[0], y_positions[0]))
        SCREEN.blit(text_titleP, (x_positions[1], y_positions[1]))
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clock.tick(60)
        pygame.display.flip()
    
    return 0

def runSimulation():
    running = True
    m=10
    currSimul = Simulation(m=m)
    currGrid = AttractiveGrid(grid=currSimul.matrix, m=m)
    gridTrigger = False
    spacePressed = False

    while running:
        currSimul.updateParticles()
        currSimul.renderScreen(SCREEN)
        
        if gridTrigger:
            currGrid.render(SCREEN)

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    spacePressed = True
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_SPACE:
                    if spacePressed:
                        spacePressed = False
                        if gridTrigger:
                            gridTrigger = False
                        else:
                            gridTrigger = True

        clock.tick(60)
        pygame.display.flip()
    
    return 1

def main():
    currScreen = 2

    while True:
        if currScreen == 0:
            break
        elif currScreen == 1:
            currScreen = menuScreen()
        elif currScreen == 2:
            currScreen = runSimulation()

if __name__ == "__main__":
    main()