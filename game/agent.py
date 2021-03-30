import threading
import settings as s
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
EXPERIENCE = namedtuple("Experience", 
        field_names=["state", "action", "reward", "next_state"])

GAMMA = 0.999

is_calc_grad = False

class LearnerAgent:

    agent_instance = None

    @staticmethod
    def create_agent_instance(game_inst):
        if LearnerAgent.agent_instance is None:
            LearnerAgent.agent_instance = LearnerAgent(game_inst, EpsilonGreedyStrategy(s.EPSILON_START, s.EPSILON_END, s.EPSILON_DECAY))

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
        self.memory = ReplayMemory(s.REPLAY_MEMORY_SIZE)
        self.current_state = self.get_game_vals()

    def decide(self):
        state = self.get_game_vals()
        rate = self.learning_strat.get_rate()
        
        decision = None
        available_actions = self.api.getAvailableActions()

        if random.random() > rate:
            with torch.no_grad():
                output = self.policy_net(state).tolist()
                best_decision = (0, -1)
                for action in available_actions:
                    if output[action.value] > best_decision[1]:
                        best_decision = (action, output[action.value])
                decision = best_decision[0]
                print("Calculated (exploitation) decision: " + str(decision))
        else:
            decision = random.choice(available_actions)
            print("Random (exploration) decision: " + str(decision))


        self.choose_action(decision)

        self.memory.add(self.current_state.unsqueeze(0), torch.tensor([[decision.value]]), torch.tensor([[self.api.getReward()]]), state.unsqueeze(0))

        if self.memory.can_provide_sample(s.REPLAY_BATCH_SIZE) and safe_batch(): 
            torch.autograd.set_detect_anomaly(True)
            toggle_safe_batch()
  
            transitions = self.memory.sample(s.REPLAY_BATCH_SIZE)
            batch = EXPERIENCE(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), device=DEVICE, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                        if s is not None])

            state_batch = torch.cat(batch.state).clone()
            action_batch = torch.cat(batch.action).clone()
            reward_batch = torch.cat(batch.reward).clone()

            state_action_values = self.policy_net(state_batch).gather(1, action_batch).clone() #this line fails to compute gradient

            next_state_values = torch.zeros(s.REPLAY_BATCH_SIZE, device=DEVICE)
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach().clone()
            
            expected_state_action_values = next_state_values * GAMMA

            for i in range(s.REPLAY_BATCH_SIZE):
                expected_state_action_values[i] = expected_state_action_values[i] + reward_batch[i][0] 
            expected_state_action_values = expected_state_action_values.unsqueeze(1)
            
            loss = F.smooth_l1_loss(state_action_values.clone(), expected_state_action_values).clone()

            optimizer = optim.RMSprop(self.policy_net.parameters())

            optimizer.zero_grad()
            loss.backward()   # BUG: this fails after a few runs
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()

            toggle_safe_batch()
        

        self.current_state = state


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

def safe_batch():
    return not is_calc_grad

def toggle_safe_batch():
    global is_calc_grad 
    is_calc_grad = not is_calc_grad

class Network(nn.Module):
    
    def __init__(self, pacmanInst):
        super(Network, self).__init__()
        self.input = nn.Linear(9, s.HIDDEN_LAYER_WIDTH)
        self.hidden = nn.Linear(s.HIDDEN_LAYER_WIDTH, s.HIDDEN_LAYER_WIDTH)
        self.output = nn.Linear(s.HIDDEN_LAYER_WIDTH, 4)
        self.squishifier = nn.ReLU()
        self.api = pacmanInst

    def forward(self, x):
        x = self.input(x)
        x = self.squishifier(x)
        x = self.hidden(x)
        x = self.squishifier(x)
        x = self.output(x) # This is the layer where problems happen
        x = self.squishifier(x)
        return x

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def add(self,state, action, reward, next_state):
        #TODO: add a push counter?
        if len(self.memory) >= self.capacity:
            self.memory.append(None)
        else:
            self.memory.append(EXPERIENCE(state,action,reward,next_state))
        
    def sample(self, batch_size):
        sampled_exp = random.sample(self.memory, batch_size)
        return sampled_exp
    
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

