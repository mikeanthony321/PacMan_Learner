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
        self.squishifier =  nn.ReLU()
        self.api = pacmanInst

    def fire():
        # make thread
        thread = threading.Thread(target=listen)
        thread.start()
    
    def listen():
        # TODO: potentially change to flag
        #  while True:
        print("new thread")    
            # gameState = api.getUpdateState()
            # if gameState == 1:

                
    def get_game_vals():
        print("AHH")

    def forward(self, x):
        x = self.input(x)
        x = self.squishifier(x)
        x = self.hidden(x)
        x = self.squishifier(x)
        x = self.output(x)
        x = self.squishifier(x)
            
        return x

print("Hi")
test_NN = LearnerAgent()
test_NN.fire()
# random_data = torch.rand(9)
# result = test_NN(random_data)
# print(result)
