import sys, math, time, threading
from api.agent_analytics_frame import AgentAnalyticsFrameAPI
from network_diagram import NeuralNetwork, Layer
from settings import *
from frame_styles import *
from PyQt5.QtWidgets import QVBoxLayout, QMainWindow, QLineEdit, QTabWidget, QPushButton, \
    QTableView, QTableWidget, QDesktopWidget, QTableWidgetItem, QWidget, QHBoxLayout, QLabel, \
    QApplication, QHeaderView, QFrame
from PyQt5.QtCore import QTimer, Qt, QSize, QPoint
from PyQt5.QtGui import QFont, QPixmap, QPainter, QBrush, QPen, QColor, QRadialGradient


class Analytics(QMainWindow):
    # __metaclass__ = AgentAnalyticsFrameAPI

    analytics_instance = None

    @staticmethod
    def create_analytics_instance(monitor_size, agent_instance):
        if Analytics.analytics_instance is None:
            Analytics.analytics_instance = Analytics(monitor_size, agent_instance)

    @staticmethod
    def update_frame():
        thread = threading.Thread(target=Analytics.analytics_instance.update)
        thread.start()

    def __init__(self, monitor_size, agent_instance):
        super().__init__()

        # Initialize the window
        self.window = QMainWindow()
        self.window.resize(WIDTH * 2, HEIGHT)
        self.window.setStyleSheet("background-color: black; color: white;")
        self.window.move(math.floor(monitor_size.width() / 2 - (1.5 * WIDTH)),
                         math.floor(monitor_size.height() / 2 - HEIGHT / 2 - 31))
        self.setWindowTitle('Pac-Man Leaner Analytics')
        self.agent_interface = agent_instance
        self.diagram_width = 450
        self.diagram_height = 350
        self.neural_network = None
        self.visualizer = None
        self.tabs = QTabWidget()

        # State Variables
        self.running = False
        self.tar_high_score = None
        self.learning_rate = None
        self.timer_min = 0
        self.timer_sec = 0
        self.timer_ms = 0
        self.table_col = 4
        self.table_rows = 5

        # Create Neural Network
        self.create_nn()

        self.initialize_structure()

        # Load the screen
        self.load_screen()

    # -- -- -- GENERAL FUNCTIONS -- -- -- #
    def create_nn(self):
        structure_array = self.agent_interface.get_network_structure()
        network = NeuralNetwork()
        for i in range(len(structure_array)):
            layer = Layer(i)
            for j in range(structure_array[i]):
                layer.add_node(None)
            network.add_layer(layer)

        self.neural_network = network

    def initialize_structure(self):
        if self.neural_network is not None:
            for i in range(len(self.neural_network.layers)):
                x_space = self.diagram_width / (len(self.neural_network.layers) + 1)
                x = ((self.diagram_width / (len(self.neural_network.layers) + 1)) * (i + 1))
                for j in range(len(self.neural_network.layers[i].nodes)):
                    y = ((self.diagram_height / (len(self.neural_network.layers[i].nodes) + 1)) * (j + 1))
                    self.neural_network.layers[i].nodes[j].set_position(x, y)

            for i in range(len(self.neural_network.layers)):
                for j in range(len(self.neural_network.layers[i].nodes)):
                    if i > 0:
                        for k in range(len(self.neural_network.layers[i - 1].nodes)):
                            self.neural_network.layers[i].nodes[j].set_connection(
                                self.neural_network.layers[i - 1].nodes[k])

            for i in range(len(self.neural_network.layers)):
                activation_vals = self.agent_interface.get_activation_vals(i)
                for j in range(len(self.neural_network.layers[i].nodes)):
                    if i > 0:
                        weights = self.agent_interface.get_weights(i - 1)
                        for k in range(len(self.neural_network.layers[i - 1].nodes)):
                            self.neural_network.layers[i].nodes[j].connections[k].set_weight(
                                0 if weights is None else weights[j][k])
                    self.neural_network.layers[i].nodes[j].set_activation_value(
                        0 if activation_vals is None else activation_vals[j])

        else:
            print("Neural Network object has not been initialized")

    def update(self):
        for i in range(len(self.neural_network.layers)):
            activation_vals = self.agent_interface.get_activation_vals(i)
            for j in range(len(self.neural_network.layers[i].nodes)):
                if i > 0:
                    weights = self.agent_interface.get_weights(i - 1)
                    for k in range(len(self.neural_network.layers[i - 1].nodes)):
                        if weights is not None:
                            self.neural_network.layers[i].nodes[j].connections[k].set_weight(weights[j][k])
                self.neural_network.layers[i].nodes[j].set_activation_value(
                    0 if activation_vals is None else activation_vals[j])

        self.visualizer.update_diagram(self.neural_network)
        self.update_table()




    def showTime(self):
        if self.running:
            self.timer_sec += 1
            if self.timer_sec >= 60:
                self.timer_min += 1
                self.timer_sec = 0

            self.formatTime()

    def formatTime(self):
        if self.timer_sec < 10:
            self.timer_label.setText("%d:0%d" % (self.timer_min, self.timer_sec))
        else:
            self.timer_label.setText("%d:%d" % (self.timer_min, self.timer_sec))

    # -- -- -- BUTTON FUNCTIONS -- -- -- #
    def beginButton(self):
        if self.tar_high_score is not None and self.learning_rate is not None:
            self.timer.start(1000)
            self.running = True
            self.agent_interface.start_sim()
            self.help_text_label.setText("")
        else:
            print("Target high score and learning rate must be set before beginning simulation")
            self.help_text_label.setText("Target High Score and Learning Rate must be set before beginning simulation")

    def highScoreButton(self):
        if self.tar_high_score_input.text() != "":
            if not self.running:
                self.tar_high_score = self.tar_high_score_input.text()
                self.tar_high_score = int(self.tar_high_score)
                self.tar_high_score_label.setText("Target High Score: " + self.tar_high_score_input.text())
                self.tar_high_score_input.setText("")
                self.help_text_label.setText("")
                self.agent_interface.set_target_score(self.tar_high_score)
            else:
                # print("You must stop the sim to enter a target high score")
                self.help_text_label.setText("The Target High Score cannot be changed while the game is running")
                self.tar_high_score_input.setText("")
        else:
            print("Invalid Entry: Please enter a target high score")

    def learningRateButton(self):
        if self.learning_rate_input.text() != "":
            if not self.running:
                self.learning_rate = self.learning_rate_input.text()
                self.learning_rate = float(self.learning_rate)
                if self.learning_rate < 1.0:
                    self.learning_rate_label.setText("Learning Rate: " + self.learning_rate_input.text())
                    self.help_text_label.setText("")
                    self.agent_interface.set_learning_rate(self.learning_rate)
                else:  # I do not know enough about learning rate to know if we want to require it be under 1, just did to have a test
                    self.help_text_label.setText("Invalid Learning Rate")
                self.learning_rate_input.setText("")
            else:
                print("You must stop the sim to enter a new learning rate")
                self.help_text_label.setText("The Learning Rate cannot be changed while the game is running")
                self.learning_rate_input.setText("")
        else:
            print("Invalid Entry: Please enter a learning rate")

    def load_screen(self):
        # Initialize layout
        self.center_widget = QWidget()
        layout = QVBoxLayout()
        self.tabs.setStyleSheet(QTAB_STYLE)

        self.tabs.addTab(self.main_tab_UI(), "Start")
        self.tabs.addTab(self.explainAI_tab_UI(), "Model Insights")
        layout.addWidget(self.tabs)
        self.center_widget.setLayout(layout)
        self.window.setCentralWidget(self.center_widget)
        self.window.show()

    def main_tab_UI(self):

        # Set up layout for main tab
        mainTab = QWidget()
        main_tab_layout = QHBoxLayout()
        left_layout = QVBoxLayout()
        right_layout = QVBoxLayout()

        # Left Panel Title
        self.setup_label = QLabel('Learning Parameters')
        self.setup_label.setStyleSheet(TITLE_STYLE)
        left_layout.addWidget(self.setup_label)
        left_layout.addSpacing(20)

        # Create the Target High Score input box and button
        hlayout1 = QHBoxLayout()
        self.tar_high_score_input = QLineEdit(self.window)
        self.tar_high_score_input.setMinimumSize(300, 30)
        self.tar_high_score_input.setStyleSheet(QLINE_STYLE)
        self.tar_high_score_button = QPushButton('Set Target High Score', self.window)
        self.tar_high_score_button.clicked.connect(self.highScoreButton)
        self.tar_high_score_button.setMinimumSize(180, 30)
        self.tar_high_score_button.setStyleSheet(BUTTON_STYLE)
        hlayout1.addWidget(self.tar_high_score_input, 1)
        hlayout1.addSpacing(5)
        hlayout1.addWidget(self.tar_high_score_button)
        left_layout.addLayout(hlayout1)
        left_layout.addSpacing(10)

        # Create the Learning Rate input box and button
        hlayout2 = QHBoxLayout()
        self.learning_rate_input = QLineEdit(self.window)
        self.learning_rate_input.setMinimumSize(300, 30)
        self.learning_rate_input.setStyleSheet(QLINE_STYLE)
        self.learning_rate_button = QPushButton('Set Learning Rate', self.window)
        self.learning_rate_button.setMinimumSize(180, 30)
        self.learning_rate_button.clicked.connect(self.learningRateButton)
        self.learning_rate_button.setStyleSheet(BUTTON_STYLE)
        hlayout2.addWidget(self.learning_rate_input)
        hlayout2.addSpacing(5)
        hlayout2.addWidget(self.learning_rate_button)
        left_layout.addLayout(hlayout2)
        left_layout.addSpacing(20)

        # Create the confirmation / help panel
        help_layout = QVBoxLayout()
        self.helpPanelWidget = BorderWidget()
        self.tar_high_score_label = QLabel("Target High Score is not set")
        help_layout.addWidget(self.tar_high_score_label)
        self.learning_rate_label = QLabel("Learning Rate is not set")
        help_layout.addWidget(self.learning_rate_label)
        self.helpPanelWidget.setLayout(help_layout)
        self.help_text_label = QLabel("")
        help_layout.addWidget(self.help_text_label)
        left_layout.addWidget(self.helpPanelWidget)
        left_layout.addSpacing(20)

        # Create the Button for the Begin button to start the game (sim)
        self.begin_button = QPushButton('Start Learning Agent', self.window)
        self.begin_button.setStyleSheet(BUTTON_STYLE)
        self.begin_button.setMinimumSize(130, 30)
        left_layout.addWidget(self.begin_button)
        self.begin_button.clicked.connect(self.beginButton)
        left_layout.addSpacing(20)
        left_layout.addStretch()

        # Right Panel Title
        self.visualization_label = QLabel('Neural Network Activity', self.window)
        self.visualization_label.setStyleSheet(TITLE_STYLE)
        right_layout.addWidget(self.visualization_label)

        # Initialize the Visualizer
        self.visualizer = Visualizer(self.neural_network, self.diagram_width, self.diagram_height, self.agent_interface)
        right_layout.addWidget(self.visualizer)
        right_layout.addSpacing(20)

        # Create and add Qvalue Table
        self.createTable()
        right_layout.addWidget(self.q_value_table)
        self.q_value_table.setMinimumSize(250, 150)
        self.set_table_dimension()
        right_layout.addSpacing(20)

        # Create the Label and Timer for the Execution Timer
        timer_layout = QHBoxLayout()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.showTime)
        time_label = QLabel('Running time')
        time_label.setStyleSheet(TEXT_STYLE)
        self.timer_label = QLabel('0:00', self.window)
        self.timer_label.setStyleSheet(TEXT_STYLE)
        self.timer_label.setAlignment(Qt.AlignRight)
        timer_layout.addWidget(time_label)
        timer_layout.addWidget(self.timer_label)
        left_layout.addLayout(timer_layout)
        left_layout.addSpacing(10)

        # Show the window with current widgets
        left_layout.setSpacing(0)

        main_tab_layout.addLayout(left_layout)
        main_tab_layout.addSpacing(30)
        main_tab_layout.addLayout(right_layout)
        main_tab_layout.addStretch()
        mainTab.setLayout(main_tab_layout)
        return mainTab

    def explainAI_tab_UI(self):
        explainAITab = QWidget()
        explainAI_tab_layout = QVBoxLayout()

        """
        Explainability implementation here
        """
        explainAITab.setLayout(explainAI_tab_layout)
        return explainAITab

    def createTable(self):
        self.q_value_table = QTableWidget(self.table_rows, self.table_col, self.window)
        self.q_value_table.setHorizontalHeaderLabels(["Up", "Down", "Left", "Right"])
        for i in range(self.table_rows):
            for j in range(self.table_col):
                self.q_value_table.setItem(i, j, QTableWidgetItem(""))

        self.q_value_table.setStyleSheet(TABLE_STYLE)

    def set_table_dimension(self):
        for i in range(0, self.table_col):
            self.q_value_table.horizontalHeader().setSectionResizeMode(i, QHeaderView.Stretch)
        for i in range(0, self.table_rows):
            self.q_value_table.verticalHeader().setSectionResizeMode(i, QHeaderView.Stretch)

    def update_table(self):

        for i in range(self.table_rows - 1):
            for j in range(self.table_col):
                self.q_value_table.item(self.table_rows - i - 1, j).setText(
                    self.q_value_table.item(self.table_rows - i - 2, j).text())

        for k in range(self.table_col):
            self.q_value_table.item(0, k).setText(
                str(round(self.neural_network.layers[4].nodes[k].get_activation_value(), 5)))

    def setRunning(self, isRunning):
        self.running = isRunning


class Visualizer(QWidget):
    def __init__(self, network_diagram, width, height, interface):
        super().__init__()
        # Initialize the diagram
        self.title = "Visualizer"
        self.width = width
        self.height = height
        self.node_size = int(self.height / 13)
        self.base_color = (51, 199, 255)
        self.color_val_param = 1
        self.thickness_param = 3
        self.base_line_thickness_param = 1

        # Set the network
        self.network = network_diagram
        self.agent_interface = interface

        # Initialize the UI
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

    def paintEvent(self, event):

        painter = QPainter(self)

        # Draw connections
        for i in range(len(self.network.layers)):
            for j in range(len(self.network.layers[i].nodes)):
                if i > 0:
                    for k in range(len(self.network.layers[i - 1].nodes)):
                        painter.setPen(self.transformConnection(self.network.layers[i].nodes[j].connections[k].weight))
                        painter.drawLine((self.network.layers[i].nodes[j].x + math.floor((self.node_size / 2))),
                                         (self.network.layers[i].nodes[j].y + math.floor((self.node_size / 2))),
                                         (self.network.layers[i - 1].nodes[k].x + math.floor((self.node_size / 2))),
                                         (self.network.layers[i - 1].nodes[k].y + math.floor((self.node_size / 2))))

        # Draw nodes
        painter.setPen(QPen(self.transformColor(-0.5, 1), 1))
        for a in range(len(self.network.layers)):
            for b in range(len(self.network.layers[a].nodes)):
                radialGradient = QRadialGradient(
                    QPoint((self.network.layers[a].nodes[b].x + math.floor(self.node_size / 2)),
                           self.network.layers[a].nodes[b].y + math.floor(self.node_size / 2)), 40)

                node_color_1 = self.transformColor(self.network.layers[a].nodes[b].get_activation_value(), 0.5)
                node_color_2 = self.transformColor(self.network.layers[a].nodes[b].get_activation_value(), -1)
                node_color_3 = self.transformColor(self.network.layers[a].nodes[b].get_activation_value(), -1.5)

                radialGradient.setColorAt(0.1, node_color_1)
                radialGradient.setColorAt(0.5, node_color_2)
                radialGradient.setColorAt(1.0, node_color_3)
                painter.setBrush(QBrush(radialGradient))
                painter.drawEllipse(self.network.layers[a].nodes[b].x, self.network.layers[a].nodes[b].y,
                                    self.node_size, self.node_size)

    def transformColor(self, val, param):
        r = max(min(math.floor((self.base_color[0] + (self.base_color[1] * val * self.color_val_param * param))), 255),
                self.base_color[0] * 0.2)
        g = max(min(math.floor((self.base_color[1] + (self.base_color[1] * val * self.color_val_param * param))), 255),
                self.base_color[0] * 0.2)
        b = max(min(math.floor((self.base_color[2] + (self.base_color[2] * val * self.color_val_param * param))), 255),
                self.base_color[0] * 0.2)
        return QColor(r, g, b)

    def transformConnection(self, weight_param):
        thickness = (self.thickness_param * weight_param) + self.base_line_thickness_param

        return QPen(QColor(90, 90, 90), thickness)

    def minimumSizeHint(self):
        return QSize(450, 350)

    def sizeHint(self):
        return QSize(500, 350)

    def update_diagram(self, network_diagram):
        self.network = network_diagram
        self.repaint()


class BorderWidget(QFrame):

    def __init__(self, *args):
        super(BorderWidget, self).__init__(*args)
        self.setStyleSheet(WIDGET_STYLE)

    def minimumSizeHint(self):
        return QSize(550, 100)
