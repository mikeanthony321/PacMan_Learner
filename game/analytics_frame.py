import sys, math, time
from network_visualizer_test import get_network_diagram, network_update
from api.agent_analytics_frame import AgentAnalyticsFrameAPI
from network_diagram import NeuralNetwork, Layer
from settings import *
from PyQt5.QtWidgets import QVBoxLayout, QMainWindow, QLineEdit, QPushButton, QTableWidget, QDesktopWidget, QTableWidgetItem, QWidget, QHBoxLayout, QLabel, QApplication
from PyQt5.QtCore import QTimer, Qt, QSize, QPoint
from PyQt5.QtGui import QFont,QPixmap, QPainter, QBrush, QPen, QColor, QRadialGradient

class Analytics(QMainWindow):
    #__metaclass__ = AgentAnalyticsFrameAPI
    def __init__(self, monitor_size):
        super().__init__()

        # Initialize the window
        self.window = QMainWindow()
        self.window.resize(WIDTH * 2, HEIGHT)
        self.window.move(math.floor(monitor_size.width() / 2 - (1.5 * WIDTH)), math.floor(monitor_size.height() / 2 - HEIGHT / 2 - 31))
        self.setWindowTitle('Pac-Man Leaner Analytics')
        self.agent_interface = AgentAnalyticsFrameAPI

        # State Variables
        self.running = False
        self.tar_high_score = 0
        self.timer_min = 0
        self.timer_sec = 0
        self.timer_ms = 0

        # Create Neural Network
        self.create_nn()

        # Load the screen
        self.load_screen()

    # -- -- -- GENERAL FUNCTIONS -- -- -- #
    def create_nn(self):
        structure_array = self.agent_interface.get_network_structure(self)
        network = NeuralNetwork()
        for i in range(len(structure_array)):
            layer = Layer(i)
            for j in range(structure_array[i]):
                layer.add_node(None)
            network.add_layer(layer)
        self.neural_network = network

    def showTime(self):
        if self.running:
            self.timer_sec += 1
            if self.timer_sec >= 60:
                self.timer_min += 1
                self.timer_sec = 0

            self.update_visualization()
            self.formatTime()

    def formatTime(self):
        if self.timer_sec < 10:
            self.timer_label.setText("Execution timer: %d:0%d" % (self.timer_min, self.timer_sec))
        else:
            self.timer_label.setText("Execution timer: %d:%d" % (self.timer_min, self.timer_sec))

    # -- -- -- BUTTON FUNCTIONS -- -- -- #
    def beginButton(self):
        self.timer.start(1000)
        self.running = True

    def highScoreButton(self):
        if not self.running:
            self.tar_high_score = self.tar_high_score_input.text()
            self.tar_high_score = int(self.tar_high_score)
            print("The target high score is: " + self.tar_high_score_input.text())
            self.tar_high_score_input.setText("")
            self.agent_interface.set_target_score(self.tar_high_score)
        else:
            print("You must stop the sim to enter a target high score")
            self.tar_high_score_input.setText("")

    def learningRateButton(self):
        if not self.running:
            self.learning_rate = self.learning_rate_input.text()
            self.learning_rate = float(self.learning_rate)
            if self.learning_rate < 1.0:
                print(self.learning_rate)
                self.agent_interface.set_learning_rate(self.learning_rate)
            else: #I do not know enough about learning rate to know if we want to require it be under 1, just did to have a test
                print('Please enter a number less than 1')
            self.learning_rate_input.setText("")
        else:
            print("You must stop the sim to enter a new learning rate")
            self.learning_rate_input.setText("")

    def load_screen(self):
        # Initialize layout
        self.center_widget = QWidget()
        layout = QVBoxLayout()

        # Create the Label/Text Input/Button for the Target High Score
        self.tar_high_score_label = QLabel('Target High Score', self.window)
        layout.addWidget(self.tar_high_score_label)
        self.tar_high_score_input = QLineEdit(self.window)
        self.tar_high_score_input.resize(150, 30)
        layout.addWidget(self.tar_high_score_input)
        self.tar_high_score_button = QPushButton('Submit', self.window)
        self.tar_high_score_button.clicked.connect(self.highScoreButton)
        layout.addWidget(self.tar_high_score_button)

        # Create the Label/Text Input/Button for the Learning Rate
        self.learning_rate_label = QLabel('Learning Rate', self.window)
        layout.addWidget(self.learning_rate_label)
        self.learning_rate_input = QLineEdit(self.window)
        self.learning_rate_input.resize(150, 30)
        layout.addWidget(self.learning_rate_input)
        self.learning_rate_button = QPushButton('Submit', self.window)
        self.learning_rate_button.clicked.connect(self.learningRateButton)
        layout.addWidget(self.learning_rate_button)

        # Create the Label/Button for the Begin button to start the game (sim)
        self.begin_label = QLabel('Begin Sim', self.window)
        layout.addWidget(self.begin_label)
        self.begin_button = QPushButton('Begin', self.window)
        layout.addWidget(self.begin_button)
        self.begin_button.clicked.connect(self.beginButton)

        # Create the Label and Timer for the Execution Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.showTime)
        self.timer_label = QLabel('Execution timer: 0:00', self.window)
        self.timer_label.resize(150, 50)
        self.timer_label.setFont(QFont('Arial', 12))
        layout.addWidget(self.timer_label)

        # Initialize the Visualizer
        self.visualizer = Visualizer(get_network_diagram(), self.agent_interface)
        self.visualizer.resize(150, 500)
        layout.addWidget(self.visualizer)

        # Show the window with current widgets
        layout.setSpacing(0)
        self.center_widget.setLayout(layout)
        self.window.setCentralWidget(self.center_widget)
        self.window.show()

class Visualizer(QWidget):
    def __init__(self, network_diagram, interface):
        super().__init__()
        # Initialize the diagram
        self.title = "Visualizer"
        self.width = 400
        self.height = 400
        self.node_size = 35
        self.color_var_param = 250
        self.base_color_param = 80
        self.thickness_param = 3
        self.base_line_thickness_param = 1

        # Set the network
        self.network = network_diagram
        self.agent_interface = interface

        # Initialize the UI
        self.initUI()
        self.initialize_structure()

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

    def initialize_structure(self):
        for i in range(len(self.network.layers)):
            x_space = self.width / (len(self.network.layers) + 1)
            x = ((self.width / (len(self.network.layers) + 1)) * (i + 1))
            for j in range(len(self.network.layers[i].nodes)):
                y = ((self.height / (len(self.network.layers[i].nodes) + 1)) * (j + 1))
                self.network.layers[i].nodes[j].set_position(x, y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.blue, 2, Qt.SolidLine))

        # Draw connections
        for i in range(len(self.network.layers)):
            for j in range(len(self.network.layers[i].nodes)):
                if i > 0:
                    for k in range(len(self.network.layers[i-1].nodes)):
                        thickness = (self.thickness_param) * (self.network.layers[i].nodes[j].connections[k].weight)\
                                    + self.base_line_thickness_param
                        painter.setPen(QPen(QColor(90, 90, 90), thickness))
                        painter.drawLine((self.network.layers[i].nodes[j].x + math.floor((self.node_size / 2))),
                                         (self.network.layers[i].nodes[j].y + math.floor((self.node_size / 2))),
                                         (self.network.layers[i-1].nodes[k].x + math.floor((self.node_size / 2))),
                                         (self.network.layers[i-1].nodes[k].y + math.floor((self.node_size / 2))))

        # Draw nodes
        painter.setPen(QPen(QColor(80, 100, 150), 1))
        for a in range(len(self.network.layers)):
            for b in range(len(self.network.layers[a].nodes)):
                radialGradient = QRadialGradient(
                    QPoint((self.network.layers[a].nodes[b].x + math.floor(self.node_size / 2)),
                           self.network.layers[a].nodes[b].y + math.floor(self.node_size / 2)), 40)

                node_color_1 = QColor(min(math.floor(self.base_color_param + self.network.layers[a].nodes[
                    b].get_activation_value() * (self.color_var_param * 1.5)), 255), min(math.floor(
                    self.base_color_param + self.network.layers[a].nodes[
                        b].get_activation_value() * (self.color_var_param * 1.3)), 255), 255)
                node_color_2 = QColor(min(math.floor(self.base_color_param + self.network.layers[a].nodes[
                    b].get_activation_value() * (self.color_var_param * 1)), 255), min(math.floor(
                    self.base_color_param + self.network.layers[a].nodes[
                        b].get_activation_value() * (self.color_var_param * 0.9)), 255), 255)
                node_color_3 = QColor(min(math.floor(self.base_color_param + self.network.layers[a].nodes[
                    b].get_activation_value() * (self.color_var_param * 0.8)), 255), min(math.floor(
                    self.base_color_param + self.network.layers[a].nodes[
                        b].get_activation_value() * (self.color_var_param * 0.6)), 255), 255)

                radialGradient.setColorAt(0.1, node_color_1)
                radialGradient.setColorAt(0.5, node_color_2)
                radialGradient.setColorAt(1.0, node_color_3)
                painter.setBrush(QBrush(radialGradient))
                painter.drawEllipse(self.network.layers[a].nodes[b].x, self.network.layers[a].nodes[b].y,
                                    self.node_size, self.node_size)

    def minimumSizeHint(self):
        return QSize(400, 400)

    def sizeHint(self):
        return QSize(400, 400)

    def update_diagram(self, network_diagram):
        self.network = network_diagram
        self.repaint()
