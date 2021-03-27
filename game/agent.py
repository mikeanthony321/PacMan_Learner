import threading
from collections import namedtuple, deque
from enum import Enum
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device("cpu") # use CPU rather than Cuda GPU to power agent
HIDDEN_LAYER_WIDTH = 5
BATCH_SIZE = 100

class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

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
        self.target_net.load_state_dict(self.policy_net.load_state_dict())
        self.target_net.eval()


    def decide(self, state):   
        state = self.get_game_vals()
        rate = self.learning_strat.get_rate()
        
        if random.random() > rate:
            with torch.no_grad():
                return policy_net(state)
        else:
            return random.choice(np.arange(self.action_size))
        # output = self.policy_net.forward(state)
        # print(str(output))
        
        # # TODO: create comparision between networks
        # criterion = torch.nn.MSELoss()
        # self.policy_net.train()
        # self.target_net.eval()

    # def choose_action(self, state):
    #     state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
    #     self.policy_net.eval()
    #     with torch.no_grad():
    #         action_values = self.policy_net(state)
    #     self.policy_net.train()

    #     #Epsilon -greedy action selction
    #     if random.random() > EPSILON_THRESHOLD:
    #         return np.argmax(action_values.cpu().data.numpy())
    #     else:
    #         return random.choice(np.arange(self.action_size))
    
    def get_game_vals(self):
        player_tuple = self.api.getPlayerGridCoords()
        ghost_tuple = self.api.getNearestGhostGridCoords()
        pellet_tuple = self.api.getNearestPelletGridCoords()
        power_tuple = self.api.getNearestPowerPelletGridCoords()
        power_active = self.api.isPowerPelletActive()

        print('Player: {}'.format(player_tuple))
        print('Ghost: {}'.format(ghost_tuple))
        print('Pellet: {}'.format(pellet_tuple))
        print('Power: {}'.format(power_tuple))
        print('Active: {}'.format(power_active))

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

class ReplayMemory():
    def __init__(self, action_size, buffer_size, seed):
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

class EpsilonGreedyStrategy():

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

