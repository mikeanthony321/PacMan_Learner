import threading
from collections import namedtuple, deque

from api.actions import Actions
import random
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device("cpu") # use CPU rather than Cuda GPU to power agent
HIDDEN_LAYER_WIDTH = 5
BATCH_SIZE = 100

class LearnerAgent:

    agent_instance = None

    @staticmethod
    def create_agent_instance(game_inst):
        if LearnerAgent.agent_instance is None:
            LearnerAgent.agent_instance = LearnerAgent(game_inst, EpsilonGreedyStrategy(1, 0.05, 200))

    @staticmethod
    def run_decision():
        thread = threading.Thread(target=LearnerAgent.agent_instance.decide)
        thread.start()

    def __init__(self, pacman_inst, learning_strat):
        self.api = pacman_inst
        self.learning_strat = learning_strat
        self.policy_net = Network(pacman_inst)
        self.target_net = Network(pacman_inst)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


    def decide(self):
        state = self.get_game_vals()
        rate = self.learning_strat.get_rate()
        
        decision = None
        available_actions = self.api.getAvailableActions()

        if random.random() > rate:
            with torch.no_grad():
                output = self.policy_net(state).tolist()
                max = (0, -1)
                for action in available_actions:
                    if output[action] > max[1]:
                        max = (action, output[action])
                decision = max[0]
        else:
            decision = random.choice(available_actions)
        
        self.choose_action(decision)
        # output = self.policy_net.forward(state)
        # print(str(output))
        
        # # TODO: create comparision between networks
        # criterion = torch.nn.MSELoss()
        # self.policy_net.train()
        # self.target_net.eval()

    def choose_action(self, decision):
        if decision is Actions.UP:
            self.api.moveUp()
        elif decision is Actions.DOWN:
            self.api.moveDown()
        elif decision is Actions.LEFT:
            self.api.moveLeft()
        elif decision is Actions.RIGHT:
            self.api.moveRight()
    
    def get_game_vals(self):
        player_tuple = self.api.getPlayerGridCoords()
        ghost_tuple = self.api.getNearestGhostGridCoords()
        pellet_tuple = self.api.getNearestPelletGridCoords()
        power_tuple = self.api.getNearestPowerPelletGridCoords()
        power_active = self.api.isPowerPelletActive()

        tensor = torch.Tensor([player_tuple[0], player_tuple[1], ghost_tuple[0], ghost_tuple[1],
                               pellet_tuple[0], pellet_tuple[1], power_tuple[0], power_tuple[1],
                               1 if power_active else 0])
        return tensor

class Network(nn.Module):
    
    def __init__(self, pacmanInst):
        super(Network, self).__init__()
        self.input = nn.Linear(9, HIDDEN_LAYER_WIDTH)
        self.hidden = nn.Linear(HIDDEN_LAYER_WIDTH, HIDDEN_LAYER_WIDTH)
        self.output = nn.Linear(HIDDEN_LAYER_WIDTH, 4)
        self.squishifier = nn.ReLU()
        self.api = pacmanInst

    def forward(self, x):
        x = self.input(x)
        x = self.squishifier(x)
        x = self.hidden(x)
        x = self.squishifier(x)
        x = self.output(x)
        x = self.squishifier(x)
        return x

class ReplayMemory:
    def __init__(self, capacity, buffer_size, seed):
        self.capacity = capacity
        self.memory = deque(maxlen=buffer_size)
        self.experiences = namedtuple("Experience", 
        field_names=["state", "action", "reward", "next_state"])
        self.seed = random.seed(seed)

    def add(self,state, action, reward, next_state):
        #TODO: add a push counter?
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        else:
            self.memory.append((state,action,reward,next_state))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def can_provide_sample(self, batch_size):
        return len(self.memory) >= batch_size

    def __len__(self):
        return len(self.memory)

class EpsilonGreedyStrategy:

    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
        self.step_count = 0
        self.threshold = 0
    
    def get_rate(self):
        self.threshold = self.end + (self.start - self.end) * math.exp(-1 * self.step_count / self.decay)
        self.step_count += 1    
        return self.threshold

