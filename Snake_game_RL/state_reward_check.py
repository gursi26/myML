'''
Normal snake game in python, takes user input.
'''

import pygame, sys, random
from pygame.math import Vector2

pygame.init()

## Fruit class
class FRUIT():
    
    def __init__(self):
        self.randomize()

    def draw_fruit(self):
        fruit_rect = pygame.Rect(int(self.pos.x) * cell_size, int(self.pos.y) * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, (255,51,51), fruit_rect)

    def randomize(self):
        self.x = random.randint(0,cell_number - 1)
        self.y = random.randint(0,cell_number - 1)
        self.pos = Vector2(self.x, self.y)


# Snake class
class SNAKE():

    def __init__(self):
        self.body = [Vector2(9,10), Vector2(8,10), Vector2(7,10)]
        self.direction = Vector2(1,0)
        self.new_block = False

    def draw_snake(self):
        for block in self.body :
            body_rect = pygame.Rect(int(block.x) * cell_size, int(block.y) * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, (102,102,255), body_rect)

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
        self.reward = 0
        self.done = False

    def update(self):
        self.reward = 0
        self.snake.move_snake()
        self.check_collision()
        self.check_fail()

    def draw_elements(self):
        self.draw_grass()
        self.draw_score()
        self.snake.draw_snake()
        self.fruit.draw_fruit()

    def check_collision(self):
        if self.fruit.pos == self.snake.body[0]:
            self.reward = 10
            self.fruit.randomize()
            self.snake.add_block()

    def check_fail(self):
        if not 0 <= self.snake.body[0].x < cell_number or not 0 <= self.snake.body[0].y < cell_number:
            self.game_over()

        for block in self.snake.body[1:] :
            if block == self.snake.body[0]:
                self.game_over()

    def game_over(self):
        self.reward = -10
        self.done = True

    def draw_grass(self):
        grass_color = (167,209,61)

        for row in range(cell_number) :

            if row % 2 == 0 : 
                alt = True
            else :
                alt = False 

            for col in range(cell_number):
                if alt : 
                    if col % 2 == 0 : 
                        grass_rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                        pygame.draw.rect(screen, grass_color, grass_rect)
                else : 
                    if col % 2 != 0 : 
                        grass_rect = pygame.Rect(col * cell_size, row * cell_size, cell_size, cell_size)
                        pygame.draw.rect(screen, grass_color, grass_rect)

    def draw_score(self):
        score_text = 'Score : ' + str(len(self.snake.body) - 3)
        score_surface = game_font.render(score_text, True, (0,0,0))

        score_x = int(cell_size * 1 + 45)
        score_y = int(cell_size * 1 + 1)

        score_rect = score_surface.get_rect(center = (score_x, score_y))
        screen.blit(score_surface, score_rect)


# Screen init
cell_size = 40
cell_number = 20
screen = pygame.display.set_mode((cell_number * cell_size, cell_number * cell_size))

# font
game_font = pygame.font.Font(None, 40)

# Clock to regulate framerate
clock = pygame.time.Clock()

SCREEN_UPDATE = pygame.USEREVENT # Custom userevent
pygame.time.set_timer(SCREEN_UPDATE,150) # triggers given event every 150ms

main_game = MAIN()

# Main game loop
def env_step() :
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        if event.type == SCREEN_UPDATE :
            main_game.update()

        # Checking if key has been pressed and which key
        if event.type == pygame.KEYDOWN :

            if event.key == pygame.K_UP :
                if main_game.snake.direction.y != 1 : 
                    main_game.snake.direction = Vector2(0,-1)
            if event.key == pygame.K_DOWN :
                if main_game.snake.direction.y != -1 :
                    main_game.snake.direction = Vector2(0,1)
            if event.key == pygame.K_RIGHT :
                if main_game.snake.direction.x != -1 :
                    main_game.snake.direction = Vector2(1,0)
            if event.key == pygame.K_LEFT :
                if main_game.snake.direction.x != 1 :
                    main_game.snake.direction = Vector2(-1,0)

    screen.fill((175,215,70))

    main_game.draw_elements()

    pygame.display.update()
    clock.tick(60)

    return main_game.reward, main_game.done


done = False
while not done :
    reward, done = env_step()
    print('Reward : ')