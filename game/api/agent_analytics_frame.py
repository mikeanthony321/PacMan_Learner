from abc import ABC, abstractmethod


class AgentAnalyticsFrameAPI(ABC):
    """
    network_structure is a list of integers, with each item in the list representing a layer
    and the value indicating the number of nodes on that layer.
    Ex [9, 5, 5, 4] is the current structure with 9 input nodes, 5 nodes on the first hidden
    layer, 5 nodes on the second hidden layer, and 4 output nodes.

    """

    # method returns a list of integers to be used in constructing the network diagram
    # called once when initializing frame
    @abstractmethod
    def get_network_structure(self):
        pass

    # method returns the activation values of the nodes in a specified layer
    # called on update
    @abstractmethod
    def get_activation_vals(self, layer_index):
        pass

    # method returns the weights connecting the specified layer
    # called on update
    @abstractmethod
    def get_weights(self, layer_index):
        pass

    # returns the logic count as an integer
    # called on update
    @abstractmethod
    def get_logic_count(self):
        pass

    # returns ghost coordinates relative to pac-man's coordinates as a list of tuples
    # called on update
    @abstractmethod
    def get_ghost_coords(self):
        pass

    # returns coordinates relative to pac-man of nearest pellet as tuple
    # called on update
    @abstractmethod
    def get_nearest_pellet_coords(self):
        pass

    # returns coordinates relative to pac-man of nearest power pellet as tuple
    # called on update
    @abstractmethod
    def get_nearest_power_pellet_coords(self):
        pass

    # returns a boolean indicating whether a power pellet  is currently active
    # called on update
    @abstractmethod
    def get_power_pellet_active_status(self):
        pass

    # returns "UP" "DOWN" "LEFT" or "RIGHT" indicating the last decision made by agent,
    # followed by "exploration" or "exploitation" indicating the type of decision made
    # called on update
    @abstractmethod
    def get_decision(self):
        pass

    # sets the learning rate of the agent
    # called once per run
    @abstractmethod
    def set_learning_rate(self, learning_rate):
        pass

    # sets the target high score to reach
    # called once per run
    @abstractmethod
    def set_target_score(self, target_score):
        pass

    # stops the learning agent and game
    @abstractmethod
    def stop_sim(self):
        pass

    # starts the learning agent
    @abstractmethod
    def start_sim(self):
        pass