import random
from collections import namedtuple
from enum import Enum

import pygame

pygame.init()
font = pygame.font.Font('fonts/Early GameBoy.ttf', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
METALLIC = (37, 111, 21)
CALPOLY = (16, 86, 31)
ALIEN = (127, 231, 6)
BLACK = (18, 18, 18)

BLOCK_SIZE = 20
FPS = 10


class SnakeGame:

    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        # init screen
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        # init game state
        self.direction = Direction.RIGHT

        self.head = Point(self.width / 2, self.height / 2)

        # snake = [(x-Head , y-Head),
        self.snake = [self.head, Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]

        self.score = 0
        self.food = None
        self.__place_food__()

    def __place_food__(self):
        x = random.randint(0, (self.width - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self.__place_food__()

    @property
    def play_step(self):
        """
        1. Collect user input
        2. Move according to user input
        3. Check if terminal
        4. Place new food, if not collected move
        5. Update frame
        6. Return game over and score

        :returns game_over, score
        """
        #
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_d:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_w:
                    self.direction = Direction.UP
                elif event.key == pygame.K_s:
                    self.direction = Direction.DOWN

        # 2
        self.__move__(self.direction)
        self.snake.insert(0, self.head)

        # 3
        game_over = False
        if self.__collision__:
            game_over = True
            return game_over, self.score

        # 4
        if self.head == self.food:
            self.score += 1
            self.__place_food__()
        else:
            self.snake.pop()

        # 5
        self.__ui__()
        self.clock.tick(FPS)
        # 6. return game over and score
        return game_over, self.score

    @property
    def __collision__(self):
        """
        Checks if hits edges
        Checks if hits itself
        :return:
        """
        if self.width - BLOCK_SIZE >= self.head.x >= 0 and self.height - BLOCK_SIZE >= self.head.y >= 0:
            if self.head in self.snake[1:]:
                return True

            return False

        return True

    def __ui__(self):

        self.screen.fill(BLACK)

        #
        for point in self.snake:
            pygame.draw.rect(self.screen, CALPOLY, pygame.Rect(point.x, point.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.screen, METALLIC, pygame.Rect(point.x + 1, point.y + 1, 18, 18))
            pygame.draw.rect(self.screen, ALIEN, pygame.Rect(point.x + 4, point.y + 4, 12, 12))

        pygame.draw.rect(self.screen, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.screen.blit(text, [0, 0])
        pygame.display.flip()

    def __move__(self, direction):
        """
        Updates coordinate of snakes head
        :rtype: object
        """
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)

    @staticmethod
    def loop():
        while True:
            game_over, score = game.play_step

            if game_over:
                break

        print('Final Score', score)


if __name__ == '__main__':
    game = SnakeGame()
    game.loop()
    # game loop

    pygame.quit()
