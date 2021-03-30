from api.agent_analytics_frame import AgentAnalyticsFrameAPI
import random

class testAgent(AgentAnalyticsFrameAPI):
    def __init__(self):
        self.structure = [9, 5, 5, 4]
        self.count = 0
        self.learning_rate = None
        self.target_score = None
        self.running = False

    def get_network_structure(self):
        return self.structure

    def get_activation_val(self, node_address):
        return random.uniform(0.01, 0.8)

    def get_weight(self, connection_address):
        return random.uniform(0.01, 0.8)

    def get_logic_count(self):
        self.count += 1
        return self.count

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_target_score(self, target_score):
        self.target_score = target_score

    def stop_sim(self):
        self.running = False

    def start_sim(self):
        self.running = True