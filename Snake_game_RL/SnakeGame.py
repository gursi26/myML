import pygame, sys, random
from pygame.math import Vector2

pygame.init()

## Fruit class
class FRUIT():
    
    def __init__(self):
        self.x = random.randint(0,cell_number - 1)
        self.y = random.randint(0,cell_number - 1)
        self.pos = Vector2(self.x, self.y)

    def draw_fruit(self):
        fruit_rect = pygame.Rect(int(self.pos.x) * cell_size, int(self.pos.y) * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (126,166,114), fruit_rect)


# Snake class
class SNAKE():

    def __init__(self):
        self.body = [Vector2(7,10), Vector2(8,10), Vector2(9,10)]
        self.direction = Vector2(1,0)

    def draw_snake(self):
        for block in self.body :
            body_rect = pygame.Rect(int(block.x) * cell_size, int(block.y) * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (0,0,255), body_rect)

    def move_snake(self):
        body_copy = self.body[:-1] # All cells except last one
        body_copy.insert(0,body_copy[0] + self.direction) # Moves head one cell in direction specified
        self.body = body_copy # Draws rest of body as it is


# Screen init
cell_size = 40
cell_number = 20
screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))

# Clock to regulate framerate
clock = pygame.time.Clock()

fruit = FRUIT()
snake = SNAKE()

SCREEN_UPDATE = pygame.USEREVENT # Custom userevent
pygame.time.set_timer(SCREEN_UPDATE,150) # triggers given event every 150ms

# Main game loop
while True :
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == SCREEN_UPDATE :
            snake.move_snake()

        # Checking if key has been pressed and which key
        if event.type == pygame.KEYDOWN :

            if event.key == pygame.K_UP :
                snake.direction = Vector2(0,-1)
            if event.key == pygame.K_DOWN :
                snake.direction = Vector2(0,1)
            if event.key == pygame.K_RIGHT :
                snake.direction = Vector2(1,0)
            if event.key == pygame.K_LEFT :
                snake.direction = Vector2(-1,0)

    screen.fill((175,215,70))

    fruit.draw_fruit()
    snake.draw_snake()

    pygame.display.update()
    clock.tick(60)