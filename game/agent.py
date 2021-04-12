import threading
import settings as s
from collections import namedtuple, deque

from api.actions import Actions
from analytics_frame import Analytics
from api.agent_analytics_frame import AgentAnalyticsFrameAPI
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

class LearnerAgent(AgentAnalyticsFrameAPI):

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
        self.ghost_list = []
        self.pellet_tuple = ()
        self.power_tuple = ()
        self.power_active = False
        self.decision_type = ""

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
                self.decision_type = "EXPLORATION"
                print("Calculated (exploitation) decision: " + str(decision))
        else:
            decision = random.choice(available_actions)
            self.decision_type = "EXPLORATION"
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
        Analytics.update_frame()


    def choose_action(self, decision):
        if decision is Actions.UP:
            self.decision = "UP"
            self.api.moveUp()
        elif decision is Actions.DOWN:
            self.decision = "DOWN"
            self.api.moveDown()
        elif decision is Actions.LEFT:
            self.decision = "LEFT"
            self.api.moveLeft()
        elif decision is Actions.RIGHT:
            self.decision = "RIGHT"
            self.api.moveRight()
    
    def get_game_vals(self):
        ghost_list = self.api.getGhostsGridCoords()
        pellet_tuple = self.api.getNearestPelletGridCoords()
        power_tuple = self.api.getNearestPowerPelletGridCoords()
        power_active = self.api.isPowerPelletActive()

        tensor = torch.Tensor([ghost_list[0][0], ghost_list[0][1], ghost_list[1][0], ghost_list[1][1],
                               ghost_list[2][0], ghost_list[2][1], ghost_list[3][0], ghost_list[3][1],
                               pellet_tuple[0], pellet_tuple[1], power_tuple[0], power_tuple[1],
                               1 if power_active else 0])
        self.ghost_list = ghost_list
        self.pellet_tuple = pellet_tuple
        self.power_tuple = power_tuple
        self.power_active = power_active
        return tensor

# -- -- -- AGENT API FUNCTIONS -- -- -- #

    def get_network_structure(self):
        return [13, 10, 10, 8, 4]

    def get_activation_vals(self, layer_index):
        try:
            return self.policy_net.full_node_dist[layer_index]
        except IndexError:
            return None

    def get_weights(self, layer_index):
        try:
            return next((weights[1] for weights in self.policy_net.full_weight_dist if weights[0] == layer_index), None)
            # return self.policy_net.full_weight_dist[layer_index]
        except IndexError:
            return None

    def get_logic_count(self):
        return 1

    def get_ghost_coords(self):
        return self.ghost_list

    def get_nearest_pellet_coords(self):
        return self.pellet_tuple

    def get_nearest_power_pellet_coords(self):
        return self.power_tuple

    def get_power_pellet_active_status(self):
        return self.power_active

    # testing
    def get_decision(self):
        return self.decision, self.decision_type

    def set_learning_rate(self, learning_rate):
        pass

    def set_target_score(self, target_score):
        pass

    def stop_sim(self):
        pass

    def start_sim(self):
        pass

def safe_batch():
    return not is_calc_grad

def toggle_safe_batch():
    global is_calc_grad 
    is_calc_grad = not is_calc_grad

class Network(nn.Module):
    
    def __init__(self, pacmanInst):
        super(Network, self).__init__()
        self.input = nn.Linear(13, 10)
        self.h1 = nn.Linear(10, 10)
        self.h2 = nn.Linear(10, 8)
        self.output = nn.Linear(8, 4)
        self.squishifier = nn.ReLU()
        self.api = pacmanInst

        self.full_node_dist = []
        self.full_weight_dist = []

    def forward(self, x):
        input_is_batch = isinstance(x.tolist()[0], list)
        node_dist = [self.squishifier(x).tolist()]
        weight_dist = []

        for index, layer in [(0, self.input), (1, self.h1), (2, self.h2), (3, self.output)]:
            x = layer(x)
            x = self.squishifier(x)
            if not input_is_batch:
                node_dist.append(x.tolist())
                weight_dist.append((index, layer.weight.tolist()))

        if not input_is_batch:
            self.full_node_dist = node_dist
            self.full_weight_dist = weight_dist

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

