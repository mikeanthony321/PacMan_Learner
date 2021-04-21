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
    # called continuously (update rate tbd)
    @abstractmethod
    def get_activation_vals(self, layer_index):
        pass

    # method returns the weights connecting the specified layer
    # called continuously (update rate tbd)
    @abstractmethod
    def get_weights(self, layer_index):
        pass

    # returns the logic count as an integer
    # called continuously (update rate tbd)
    @abstractmethod
    def get_logic_count(self):
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

    # starts the learning agent and game
    @abstractmethod
    def start_sim(self):
        pass
