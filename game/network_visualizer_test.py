import random, time
from network_diagram import NeuralNetwork, Layer

# generates a random nn structure
def create_random_nn():
    network = NeuralNetwork()
    for i in range(random.randrange(2, 6)):
        layer = Layer(i)
        for j in range(random.randrange(3, 8)):
            layer.add_node(None)
        network.add_layer(layer)
    return network

# generates a nn based on a structure array
# structure_array: list of values where each item in the list represents a layer containing n nodes [9, 5, 5, 4]
def create_set_nn(structure_array):
    network = NeuralNetwork()
    for i in range(len(structure_array)):
        layer = Layer(i)
        for j in range(structure_array[i]):
            layer.add_node(None)
        network.add_layer(layer)
    return network



# this would be replaced with a function that accesses the weight of a connection of the neural network
def get_weight(layer1_index, node1_index, layer2_index, node2_index, previous_weight):

    # access the weight of the connection between the two nodes and return the value
    if previous_weight == None:
        return random.random()
    else:
        return (previous_weight * 0.8) + (random.random() * 0.2)


# this would be replaced with a method that accesses the q values of the learning agent
def get_qvalue(layer_index, node_index, previous_qval):

    # access the qvalue of the node
    if previous_qval == None:
        return random.uniform(0.01, 0.8)
    else:
        return (previous_qval * 0.8) + (random.uniform(0.01, 0.8) * 0.2)


def initialize(network):

    for i in range(len(network.layers)):
        for j in range(len(network.layers[i].nodes)):
            if i > 0:
                for k in range(len(network.layers[i - 1].nodes)):
                    network.layers[i].nodes[j].set_connection(network.layers[i - 1].nodes[k])


def network_update(network):
    for i in range(len(network.layers)):
        for j in range(len(network.layers[i].nodes)):
            if i > 0:
                for k in range(len(network.layers[i - 1].nodes)):
                    previous_w = network.layers[i].nodes[j].connections[k].get_weight()
                    weight = get_weight(i, j, (i - 1), k, previous_w)
                    network.layers[i].nodes[j].connections[k].set_weight(weight)
            previous_qval = network.layers[i].nodes[j].get_activation_value()
            qval = get_qvalue(i, j, previous_qval)
            network.layers[i].nodes[j].set_activation_value(qval)

# used in testing the analytics frame
def get_network_diagram():
    p = create_set_nn([9, 5, 5, 4])
    p.list_structure()
    initialize(p)
    network_update(p)
    return p