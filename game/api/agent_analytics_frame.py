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
    qvals is a nested list of float values corresponding to the q values of each node of the 
    network, organized by layer.
    Ex [[0.1, 0, 0.2], [0.1, 0.1], [0, 0.2, 0, 0.3]] 
    represents the q values of each node in the network, starting with the input layer with 
    3 nodes, then the hidden layer with 2 nodes, and output layer with 4 nodes.
    The order in which the q values of nodes within a layer is provided as
    [[(n1,1), (n1,2), (n1,3)], [(n2,1), (n2,2)], [(n3,1), (n3,2), (n3,3), (n3,4)]]
    and is consistent with the order in which connection weights are accessed in get_weights
    """

    # method returns a nested list of float values used to update the network diagram
    # called continuously (update rate tbd)
    # accesses q values of each node of the network
    @abstractmethod
    def get_qvals(self):
        pass

    """
    weights is a nested list of float values corresponding to the weights between each node of the 
    network, organized by layer, then by node.
    Ex [[[0.3, 0.2], [0.1, 0.3], [0.4, 0]], [[0.3, 0, 0, 0.1], [0, 0, 0.4, 0.2]]]
    represents the weights of a network with three layers, having three inputs, two nodes in the 
    hidden layer, and four output nodes. Weights are enumerated as 
    [[[(n1,1 -> n2,1), (n1,1 -> n2,2)], [(n1,2 -> n2,1), (n1,2 -> n2,2)]  . . .  (n2,2 -> n3,4)]]]
    and is consistent with the order in which q values are accessed in get_qvals
    """

    # method returns a nested list of float values used to update the network diagram
    # called continuously (update rate tbd)
    # accesses the connection weights of each node to every node on the following layer
    @abstractmethod
    def get_weights(self):
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
