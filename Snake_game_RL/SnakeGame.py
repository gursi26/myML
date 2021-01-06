import pygame, sys, random
from pygame.math import Vector2

pygame.init()

def rect_with_border(border, xpos, ypos, cellsize, color_inside, color_border):
    rect1 = pygame.Rect(xpos * cellsize, ypos * cellsize, cellsize, cellsize)
    rect2 = pygame.Rect(xpos * (cellsize - border/2), ypos * (cellsize- border/2), cellsize - border, cellsize - border)
    pygame.draw.rect(screen, color_border, rect1)
    pygame.draw.rect(screen, color_inside, rect2)

## Fruit class
class FRUIT():
    
    def __init__(self):
        self.randomize()

    def draw_fruit(self):
        rect_with_border(2, int(self.pos.x), int(self.pos.y), cell_size, (255,166,114), (126,166,114))

    def randomize(self):
        self.x = random.randint(0,cell_number - 1)
        self.y = random.randint(0,cell_number - 1)
        self.pos = Vector2(self.x, self.y)


# Snake class
class SNAKE():

    def __init__(self):
        self.body = [Vector2(7,10), Vector2(8,10), Vector2(9,10)]
        self.direction = Vector2(1,0)
        self.new_block = False

    def draw_snake(self):
        for block in self.body :
            body_rect = pygame.Rect(int(block.x) * cell_size, int(block.y) * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (0,0,255), body_rect)

    def move_snake(self):

        if self.new_block == True :
            body_copy = self.body[:] 
            body_copy.insert(0,body_copy[0] + self.direction) 
            self.body = body_copy 
            self.new_block = False

        else :
            body_copy = self.body[:-1] # All cells except last one
            body_copy.insert(0,body_copy[0] + self.direction) # Moves head one cell in direction specified
            self.body = body_copy # Draws rest of body as it is

    def add_block(self):
        self.new_block = True


# Game logic class
class MAIN():

    def __init__(self):
        self.snake = SNAKE()
        self.fruit = FRUIT()

    def update(self):
        self.snake.move_snake()
        self.check_collision()

    def draw_elements(self):
        self.snake.draw_snake()
        self.fruit.draw_fruit()

    def check_collision(self):

        if self.fruit.pos == self.snake.body[0]:
            self.fruit.randomize()
            self.snake.add_block()


# Screen init
cell_size = 40
cell_number = 20
screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))

# Clock to regulate framerate
clock = pygame.time.Clock()

SCREEN_UPDATE = pygame.USEREVENT # Custom userevent
pygame.time.set_timer(SCREEN_UPDATE,150) # triggers given event every 150ms

main_game = MAIN()

# Main game loop
while True :
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == SCREEN_UPDATE :
            main_game.update()

        # Checking if key has been pressed and which key
        if event.type == pygame.KEYDOWN :

            if event.key == pygame.K_UP :
                main_game.snake.direction = Vector2(0,-1)
            if event.key == pygame.K_DOWN :
                main_game.snake.direction = Vector2(0,1)
            if event.key == pygame.K_RIGHT :
                main_game.snake.direction = Vector2(1,0)
            if event.key == pygame.K_LEFT :
                main_game.snake.direction = Vector2(-1,0)

    screen.fill((175,215,70))

    main_game.draw_elements()

    pygame.display.update()
    clock.tick(60)