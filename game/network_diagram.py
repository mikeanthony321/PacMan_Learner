
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
    def __init__(self, node):
        self.link = node
        self.weight = None

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

    def set_connection(self, node):
        c = Connection(node)
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




