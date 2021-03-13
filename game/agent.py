import threading
from enum import Enum
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device("cpu") # use CPU rather than Cuda GPU to power agent
HIDDEN_LAYER_WIDTH = 5
class Actions(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class LearnerAgent():

    def __init__(self, pacmanInst):
        state = self.get_game_vals()
        self.policy_net = Network(pacmanInst)
        self.target_net = Network(pacmanInst)
        self.api = pacmanInst

    def fire(self):
        # make thread
        thread = threading.Thread(target=self.listen)
        thread.start()
    

    def listen(self):
        # TODO: potentially change to flag
        print("listen")
        has_run = False
        while not has_run: 
            gameState = self.api.getUpdateState()
            if gameState == 1:
                state = self.get_game_vals()

                criterion = torch.nn.MSELoss()
                self.policy_net.train()
                self.target_net.eval()

                print("yay")
            has_run = True    

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        #Epsilon -greedy action selction
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def get_game_vals(self):
        # TODO: replce dummy tensor with actual data
        # player_tuple = self.api.getPlayerGridCoords()
        # ghost_tuple = self.api.getNearestGhostGridCoords()
        # pellet_tuple = self.api.getNearestPelletGridCoords()
        # power_tuple = self.api.getNearestPowerPelletGridCoords()
        # power_active = self.api.isPowerPelletActive()

        # tensor = [player_tuple[0], player_tuple[1], ghost_tuple[0], ghost_tuple[1], 
        # pellet_tuple[0], pellet_tuple[1], power_tuple[0], power_tuple[1], 
        # 1 if power_active else 0]

        # tensor = torch.Tensor(tensor)
        tensor = torch.rand(9)
        return tensor

class Network(nn.Module):
    
    def __init__(self, pacmanInst):
        super(Network, self).__init__()
        self.input = nn.Linear(9, HIDDEN_LAYER_WIDTH)
        self.hidden = nn.Linear(HIDDEN_LAYER_WIDTH, HIDDEN_LAYER_WIDTH)
        self.output = nn.Linear(HIDDEN_LAYER_WIDTH, 4)
        self.squishifier = nn.ReLU()
        self.api = pacmanInst


    def fire(self):
        # make thread
        thread = threading.Thread(target=self.listen)
        thread.start()

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
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.experiences = namedtuple("Experience", 
        field_names=["state", "action", "reward", "next_state"])
        self.seed = random.seed(seed)

    def add(self,state, action, reward, next_state):
        self.memory.append((state,action,reward,next_state))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

