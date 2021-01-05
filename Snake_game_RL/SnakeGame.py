import pygame, random
from enum import Enum
from collections import namedtuple

pygame.init()

BLOCK_SIZE = 20
Point = namedtuple('Point', ('x', 'y'))
# Allows accessing elements of a tuple by name rather than index. Makes code readable

class Direction(Enum):
    # prevents errors when entering direction
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SnakeGame():

    def __init__(self, width = 640, height = 480):
        self.w = width
        self.h = height
        
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game')
        # Create the screen of specified dimensions and give it a title

        self.clock = pygame.time.Clock()
        # Game clock to control speed

        self.direction = Direction.RIGHT
        # Initializes starting direction of snake

        self.head = Point(self.w/2, self.h/2)
        # Holds the initial position of snake head

        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y), Point(self.head.x-(2 * BLOCK_SIZE),self.head.y)]

        self.score = 0
        self.food = None
        self.place_food()




    




    def place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE)//BLOCK_SIZE) * BLOCK_SIZE


    def play_step(self):
        pass



if __name__ == '__main__':

    game = SnakeGame()

    while True :
         game.play_step()



    pygame.quit()