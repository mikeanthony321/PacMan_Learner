import threading

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device("cpu") # use CPU rather than Cuda GPU to power agent
HIDDEN_LAYER_WIDTH = 5

class LearnerAgent(nn.Module):

    def __init__(self, pacmanInst):
        super(LearnerAgent, self).__init__()
        self.input = nn.Linear(9, HIDDEN_LAYER_WIDTH)
        self.hidden = nn.Linear(HIDDEN_LAYER_WIDTH, HIDDEN_LAYER_WIDTH)
        self.output = nn.Linear(HIDDEN_LAYER_WIDTH, 4)
        self.squishifier = nn.ReLU()
        self.api = pacmanInst

    def fire(self):
        # make thread
        thread = threading.Thread(target=self.listen)
        thread.start()
    
    def listen(self):
        # TODO: potentially change to flag
        has_run = False
        while not has_run: 
            gameState = self.api.getUpdateState()
            if gameState == 1:
                state = self.get_game_vals()
                output = self.forward(state)
                print(output)
            has_run = True

                
    def get_game_vals(self):
        player_tuple = self.api.getPlayerGridCoords()
        ghost_tuple = self.api.getNearestGhostGridCoords()
        pellet_tuple = self.api.getNearestPelletGridCoords()
        power_tuple = self.api.getNearestPowerPelletGridCoords()
        power_active = self.api.isPowerPelletActive()

        tensor = [player_tuple[0], player_tuple[1], ghost_tuple[0], ghost_tuple[1], 
        pellet_tuple[0], pellet_tuple[1], power_tuple[0], power_tuple[1], 
        1 if power_active else 0]

        tensor = torch.Tensor(tensor)
        return tensor

    def forward(self, x):
        x = self.input(x)
        x = self.squishifier(x)
        x = self.hidden(x)
        x = self.squishifier(x)
        x = self.output(x)
        x = self.squishifier(x)
            
        return x

# random_data = torch.rand(9)
# result = test_NN(random_data)
# print(result)
