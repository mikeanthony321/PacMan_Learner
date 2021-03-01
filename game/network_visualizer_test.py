import random
from network_diagram import NeuralNetwork, Layer


def create_random_nn():
    network = NeuralNetwork()
    for i in range(random.randrange(2, 6)):
        layer = Layer(i)
        for j in range(random.randrange(3, 8)):
            layer.add_node(0)
        network.add_layer(layer)
    return network

def create_set_nn():
    network = NeuralNetwork()
    layer0 = Layer(0)
    layer1 = Layer(1)
    layer2 = Layer(2)
    layer3 = Layer(3)
    for i in range(9):
        layer0.add_node(0)
    for j in range(5):
        layer1.add_node(0)
        layer2.add_node(0)
    for k in range(4):
        layer3.add_node(0)
    network.add_layer(layer0)
    network.add_layer(layer1)
    network.add_layer(layer2)
    network.add_layer(layer3)

    return network

# this would be replaced with a function that accesses the weight of a connection of the neural network
def get_weight(layer1_index, node1_index, layer2_index, node2_index):

    # access the weight of the connection between the two nodes and return the value
    return random.random()


# this would be replaced with a method that accesses the q values of the learning agent
def get_qvalue(layer_index, node_index):

    # access the qvalue of the node
    return random.uniform(0.01, 0.8)


def update(network):

    for i in range(len(network.layers)):
        for j in range(len(network.layers[i].nodes)):
            if i > 0:
                for k in range(len(network.layers[i - 1].nodes)):
                    weight = get_weight(i, j, (i - 1), k)
                    network.layers[i].nodes[j].set_connection(network.layers[i - 1].nodes[k], weight)
            qval = get_qvalue(i, j)
            network.layers[i].nodes[j].set_activation_value(qval)


def get_network_diagram():
    p = create_set_nn()
    p.list_structure()
    update(p)
    return p