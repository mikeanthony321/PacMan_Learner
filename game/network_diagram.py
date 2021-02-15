
from PyQt5.QtGui import QPainter, QBrush, QPen, QColor
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow

import sys, time, random


#
# Classes used to represent neural network structure, hold qvalues, weights, and xy positions in visualization
#

# Represents a connection between two nodes in the diagram
# self.link is a backward link to a node in the previous layer
class Connection():
    def __init__(self, node, weight):
        self.link = node
        self.weight = weight

    def get_weight(self):
        return self.weight

    def set_weight(self, weight):
        self.weight = weight


class Node():
    def __init__(self, activation_value, layer, node_index):
        self.activation_value = activation_value
        self.layer = layer
        self.index = node_index
        self.x = 0
        self.y = 0
        self.connections = []

    def get_activation_value(self):
        return self.activation_value

    def set_activation_value(self, val):
        self.activation_value = val

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def set_connection(self, node, weight):
        c = Connection(node, weight)
        self.connections.append(c)


class Layer():
    def __init__(self, layer_index):
        self.nodes = []
        self.index = layer_index

    def add_node(self, activation_value):
        n = Node(activation_value, self.index, len(self.nodes))
        self.nodes.append(n)

    def get_node(self, index):
        return self.nodes[index]


class NeuralNetwork():
    def __init__(self):
        self.layers = []


    def add_layer(self, layer):
        self.layers.append(layer)

    def get_layer(self, index):
        return self.layers[index]

    def list_structure(self):
        node_list = []
        layer_list = []
        for layer in self.layers:
            for node in layer.nodes:
                node_list.append(node.activation_value)
            layer_list.append(node_list)
            node_list = []
        return layer_list



class Window(QMainWindow):

    def __init__(self, neural_network):

        super().__init__()
        self.title = "Demo Window"
        self.top= 150
        self.left= 150
        self.width = 500
        self.height = 500
        self.node_size = 50
        self.color_var_param = 250
        self.base_color_param = 100
        self.thickness_param = 4
        self.base_line_thickness_param = 1
        self.network = neural_network
        self.InitWindow()
        self.initialize_structure()

    def InitWindow(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.top, self.left, self.width, self.height)
        self.show()

    def initialize_structure(self):
        for i in range(len(self.network.layers)):
            x_space = self.width/(len(self.network.layers) + 1)
            x = ((self.width / (len(self.network.layers) + 1)) * (i + 1))
            for j in range(len(self.network.layers[i].nodes)):
                y = ((self.height / (len(self.network.layers[i].nodes) + 1)) * (j + 1))
                self.network.layers[i].nodes[j].set_position(x, y)


    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))
        for i in range(len(self.network.layers)):
            for j in range(len(self.network.layers[i].nodes)):
                if i > 0:
                    for k in range(len(self.network.layers[i - 1].nodes)):
                        thickness = (self.thickness_param) * (
                            self.network.layers[i].nodes[j].connections[k].weight) + self.base_line_thickness_param
                        painter.setPen(QPen(QColor(60, 60, 60), thickness))
                        painter.drawLine((self.network.layers[i].nodes[j].x + (self.node_size / 2)),
                                     (self.network.layers[i].nodes[j].y + (self.node_size / 2)),
                                     (self.network.layers[i - 1].nodes[k].x + (self.node_size / 2)),
                                     (self.network.layers[i - 1].nodes[k].y + (self.node_size / 2)))
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        for a in range(len(self.network.layers)):
            for b in range(len(self.network.layers[a].nodes)):
                painter.setBrush(QBrush(QColor(min(self.base_color_param + self.network.layers[a].nodes[
                    b].get_activation_value() * self.color_var_param, 255),
                                               min(self.base_color_param + self.network.layers[a].nodes[
                                                   b].get_activation_value() * self.color_var_param, 255),
                                               255), Qt.SolidPattern))
                painter.drawEllipse(self.network.layers[a].nodes[b].x, self.network.layers[a].nodes[b].y,
                                    self.node_size, self.node_size)

    def update_diagram(self):
        self.update()


#
# Tests
#

def create_random_nn():
    network = NeuralNetwork()
    for i in range(random.randrange(2, 6)):
        layer = Layer(i)
        for j in range(random.randrange(3, 8)):
            layer.add_node(0)
        network.add_layer(layer)
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





p = create_random_nn()
p.list_structure()
update(p)

App = QApplication(sys.argv)

window = Window(p)

sys.exit(App.exec())

