import torch
import random
import numpy as np
from collections import deque
from snake import AIGame, Direction, Point, BLOCK_SIZE
from model import L_QNET, QTrainer
from plotter import plot
import pygame

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001


class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        # Discount rate must be <1
        self.gamma = 0.8
        self.memory = deque(maxlen = MAX_MEMORY)
          
        self.model = L_QNET(11, 350, 3)
        self.trainer = QTrainer(self.model, lr = LR, gamma=self.gamma)
        # TODO: model, trainer

    def get_state(self, game):
        head = game.snake[0]

        p_left = Point(head.x - BLOCK_SIZE, head.y)
        p_right = Point(head.x + BLOCK_SIZE, head.y)
        p_up = Point(head.x, head.y - BLOCK_SIZE)
        p_down = Point(head.x, head.y + BLOCK_SIZE)

        dir_left = game.direction ==  Direction.LEFT
        dir_right = game.direction ==  Direction.RIGHT
        dir_up = game.direction ==  Direction.UP
        dir_down = game.direction ==  Direction.DOWN
        
        state = [
            # Danger ahead
            (dir_right and game.collision(p_right)) or
            (dir_left and game.collision(p_left)) or
            (dir_up and game.collision(p_up)) or
            (dir_down and game.collision(p_down)),
            
            # Danger right
            (dir_up and game.collision(p_right)) or
            (dir_down and game.collision(p_left)) or
            (dir_left and game.collision(p_up)) or
            (dir_right and game.collision(p_down)),
            
            # Danger left
            (dir_down and game.collision(p_right)) or
            (dir_up and game.collision(p_left)) or
            (dir_right and game.collision(p_up)) or
            (dir_left and game.collision(p_down)),
            
            # Move direction
            dir_left,
            dir_right,
            dir_up,
            dir_down,
            
            # Food location
            game.food.x < game.head.x, 
            game.food.x > game.head.x, 
            game.food.y < game.head.y, 
            game.food.y > game.head.y, 
        ]
        
        return np.array(state, dtype=int) 
        
        

    def remember(self, state, action, reward, next, game_over):
       
        self.memory.append((state, action, reward, next, game_over))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:

            small_sample = random.sample(self.memory, BATCH_SIZE)

        else:
            small_sample = self.memory
        
        states, actions, rewards, next_states, game_overs = zip(*small_sample)
        self.trainer.train_step(states, actions, rewards, next_states, game_overs)

            
    def train_short_memory(self, state, action, reward, next, game_over):
        self.trainer.train_step(state, action, reward, next, game_over)

    def get_action(self, state):

        FOO = self.epsilon
        self.epsilon = FOO - self.n_games
        final_move =[0, 0, 0]
        if random.randint(0, 150) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            predicition = self.model(state0)
            move = torch.argmax(predicition).item()
            final_move[move] = 1
            
        return final_move
            
def train():
    plot_score = []
    plot_average_score = []
    plot_total_score = 0
    best_score = 0
    agent = Agent()
    game = AIGame()
    while True:

        pygame.display.set_caption('Games: ' + str(agent.n_games))

        old_state = agent.get_state(game)

   
        final_move = agent.get_action(old_state)

        reward, done, score = game.play_step(final_move)
        new_state = agent.get_state(game)

        agent.train_short_memory(old_state, final_move, reward, new_state, done)

        agent.remember(old_state, final_move, reward, new_state, done)

        if done:
       
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > best_score:
                record = score

                agent.model.save()

            plot_score.append(score)
            plot_total_score += score
            average_score = plot_total_score / agent.n_games
            plot_average_score.append(average_score)
            plot(plot_score, plot_average_score)
if __name__ == '__main__':
    train()
