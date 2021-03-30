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

    """
    node_address is a tuple representing the node to be accessed
    Ex (3, 5) is layer 3, node 5
    """

    # method returns the activation value of the specified node
    # called continuously (update rate tbd)
    # accesses activation values of each node of the network
    @abstractmethod
    def get_activation_val(self, node_address):
        pass

    """
    connection_address is a list of tuples representing the connection to be accessed
    Ex [(2, 2), (3, 4)] is the connection from layer 2, node 2, to the next layer 3, node 4
    """

    # method returns the weight of the specified connection
    # called continuously (update rate tbd)
    # accesses the connection weights of each node to every node on the following layer
    @abstractmethod
    def get_weight(self, connection_address):
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

    # starts the learning agent
    @abstractmethod
    def start_sim(self):
        pass