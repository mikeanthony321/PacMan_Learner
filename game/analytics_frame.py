import sys, math, time, threading

from PyQt5 import QtCore

from api.agent_analytics_frame import AgentAnalyticsFrameAPI
from network_diagram import NeuralNetwork, Layer
from settings import *
from frame_styles import *
from frame_text import *
from PyQt5.QtWidgets import QGridLayout, QVBoxLayout, QMainWindow, QLineEdit, QTabWidget, QPushButton, \
    QTableView, QTableWidget, QDesktopWidget, QTableWidgetItem, QWidget, QCheckBox, QRadioButton, QHBoxLayout, QLabel, \
    QApplication, QSlider, QHeaderView, QFrame, QSizePolicy, QScrollArea, QToolButton
from PyQt5.QtCore import QTimer, Qt, QSize, QPoint, QParallelAnimationGroup, QAbstractAnimation, QPropertyAnimation
from PyQt5.QtGui import QFont, QPixmap, QPainter, QBrush, QPen, QColor, QRadialGradient
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg


class Analytics(QMainWindow):
    analytics_instance = None
    running_state = False

    @staticmethod
    def create_analytics_instance(monitor_size, agent_instance):
        if Analytics.analytics_instance is None:
            Analytics.analytics_instance = Analytics(monitor_size, agent_instance)

    @staticmethod
    def update_frame():
        thread = threading.Thread(target=Analytics.analytics_instance.update)
        thread.start()

    @staticmethod
    def get_running_state():
        return Analytics.running_state

    @staticmethod
    def non_user_stop():
        Analytics.analytics_instance.stopButton()

    def __init__(self, monitor_size, agent_instance):
        super().__init__()

        # Initialize the window
        self.window = QMainWindow()
        self.window.resize(WIDTH * 2, HEIGHT)
        self.window.setStyleSheet("background-color: black; color: white;")
        self.window.move(math.floor(monitor_size.width() / 2 - (1.5 * WIDTH) - 24),
                         math.floor(monitor_size.height() / 2 - HEIGHT / 2 - 31))
        self.setWindowTitle('Pac-Man Leaner Analytics')
        self.agent_interface = agent_instance
        self.diagram_width = 300
        self.diagram_height = 350
        self.vis_tab_x_scale = 2
        self.vis_tab_y_scale = 1.5
        self.neural_network = None
        self.visualizer = None
        self.tabs = QTabWidget()

        # Network labels
        self.input_labels = ['Blinky X', 'Blinky Y', 'Inky X', 'Inky Y', 'Pinky X', 'Pinky y',
                             'Clyde X', 'Clyde Y', 'Nearest Pellet X', 'Nearest Pellet Y',
                             'Power Pellet X', 'Power Pellet Y', 'Active Power Pellet']
        self.output_labels = ['Up', 'Down', 'Left', 'Right']

        self.input_qlabels = []

        # State Variables
        self.running = False
        self.tar_high_score = None
        self.learning_rate = None
        self.timer_min = 0
        self.timer_sec = 0
        self.timer_ms = 0
        self.table_col = 4
        self.table_rows = 5

        # Plot variables
        self.start_plot = False
        self.ghosts = []
        self.nearest_pellet = []
        self.nearest_p_pellet = []
        self.decision = ""
        self.decision_type = ""
        self.decision_count = 0
        self.p_active = False
        self.up_plot = PlotStruct()
        self.down_plot = PlotStruct()
        self.left_plot = PlotStruct()
        self.right_plot = PlotStruct()

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
                x = (((self.diagram_width + 110) / (len(self.neural_network.layers) + 1)) * (i + 1) - 65)
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

        if self.agent_interface.get_logic_count() != self.decision_count:
            self.decision_count = self.agent_interface.get_logic_count()
            self.decision, self.decision_type = self.agent_interface.get_decision()
            # update
            if self.decision_count >= 2:
                if self.decision == 'UP':
                    self.up_plot.update_points(self.ghosts, self.nearest_pellet, self.nearest_p_pellet, self.p_active,
                                               self.decision_type)
                    if self.tabs.currentIndex() == 1:
                        self.up_plot.update_plot()
                elif self.decision == 'DOWN':
                    self.down_plot.update_points(self.ghosts, self.nearest_pellet, self.nearest_p_pellet, self.p_active,
                                                 self.decision_type)
                    if self.tabs.currentIndex() == 1:
                        self.down_plot.update_plot()
                elif self.decision == 'LEFT':
                    self.left_plot.update_points(self.ghosts, self.nearest_pellet, self.nearest_p_pellet, self.p_active,
                                                 self.decision_type)
                    if self.tabs.currentIndex() == 1:
                        self.left_plot.update_plot()
                elif self.decision == 'RIGHT':
                    self.right_plot.update_points(self.ghosts, self.nearest_pellet, self.nearest_p_pellet,
                                                  self.p_active,
                                                  self.decision_type)
                    if self.tabs.currentIndex() == 1:
                        self.right_plot.update_plot()

                """
                if self.tabs.currentIndex() == 2:
                    self.update_network_tab(
                        self.ghosts, self.nearest_pellet, self.nearest_p_pellet, self.p_active, self.decision)
                """

            self.ghosts = self.agent_interface.get_ghost_coords()
            self.nearest_pellet = self.agent_interface.get_nearest_pellet_coords()
            self.nearest_p_pellet = self.agent_interface.get_nearest_power_pellet_coords()
            self.p_active = self.agent_interface.get_power_pellet_active_status()

            if self.tabs.currentIndex() == 0:
                self.visualizer.update_diagram(self.neural_network)
            #elif self.tabs.currentIndex() == 2:
                #self.tab_vis.update_diagram(self.neural_network)

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
            self.timer2_label.setText("%d:0%d" % (self.timer_min, self.timer_sec))
            #self.timer3_label.setText("%d:0%d" % (self.timer_min, self.timer_sec))
        else:
            self.timer_label.setText("%d:%d" % (self.timer_min, self.timer_sec))
            self.timer2_label.setText("%d:%d" % (self.timer_min, self.timer_sec))
            #self.timer3_label.setText("%d:%d" % (self.timer_min, self.timer_sec))

    # -- -- -- BUTTON FUNCTIONS -- -- -- #
    def beginButton(self):
        if self.tar_high_score is not None and self.learning_rate is not None:
            self.timer.start(1000)
            self.running = True
            Analytics.running_state = True
            self.agent_interface.start_sim()
            self.help_text_label.setText("")
        else:
            print("Target high score and learning rate must be set before beginning simulation")
            self.help_text_label.setText("Target High Score and Learning Rate must be set before beginning simulation")

    def stopButton(self):
        if(self.running):
            self.timer_ms = 0
            self.timer_sec = 0
            self.timer_min = 0
            self.tar_high_score = None
            self.learning_rate = None
            self.tar_high_score_label.setText("Target High Score: " + self.tar_high_score_input.text())
            self.learning_rate_label.setText("Learning Rate: ")
            self.running = False
            Analytics.running_state = False

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
        if not self.running:
            self.learning_rate = float(self.learning_rate_slider.value() / 1000000)
            if self.learning_rate < 1.0:
                self.learning_rate_label.setText("Learning Rate: " + "{:.6f}".format(self.learning_rate))
                self.help_text_label.setText("")
                self.agent_interface.set_learning_rate(self.learning_rate)
            else:  # I do not know enough about learning rate to know if we want to require it be under 1, just did to have a test
                self.help_text_label.setText("Invalid Learning Rate")
        else:
            print("You must stop the sim to enter a new learning rate")
            self.help_text_label.setText("The Learning Rate cannot be changed while the game is running")

    def slider_valuechange(self):
        if not self.running:
            learning_val = float(self.learning_rate_slider.value() / 1000000)
            self.learning_rate_label.setText("Learning Rate: " + "{:.6f}".format(learning_val))

    def on_position_changed(self, p):
        for i in range(len(self.neural_network.layers)):
            x = ((((self.diagram_width + 110) / (len(self.neural_network.layers) + 1)) * (
                        i + 1) - 65) * self.vis_tab_x_scale)
            for j in range(len(self.neural_network.layers[i].nodes)):
                y = (((self.diagram_height / (len(self.neural_network.layers[i].nodes) + 1)) * (
                            j + 1)) * self.vis_tab_y_scale)
                if x + int(100 / 6) >= p.x() > x - int(100 / 6) and y + int(100 / 6) >= p.y() > y - int(100 / 6):
                    if i == 0:
                        if j == 0:
                            string = "Input Layer: Blinky X" + "\nActivation Value: {:.6f}"
                        elif j == 1:
                            string = "Input Layer: Blinky Y" + "\nActivation Value: {:.6f}"
                        elif j == 2:
                            string = "Input Layer: Inky X" + "\nActivation Value: {:.6f}"
                        elif j == 3:
                            string = "Input Layer: Inky Y" + "\nActivation Value: {:.6f}"
                        elif j == 4:
                            string = "Input Layer: Pinky X" + "\nActivation Value: {:.6f}"
                        elif j == 5:
                            string = "Input Layer: Pinky Y" + "\nActivation Value: {:.6f}"
                        elif j == 6:
                            string = "Input Layer: Clyde X" + "\nActivation Value: {:.6f}"
                        elif j == 7:
                            string = "Input Layer: Clyde Y" + "\nActivation Value: {:.6f}"
                        elif j == 8:
                            string = "Input Layer: Nearest Pellet X" + "\nActivation Value: {:.6f}"
                        elif j == 9:
                            string = "Input Layer: Nearest Pellet Y" + "\nActivation Value: {:.6f}"
                        elif j == 10:
                            string = "Input Layer: Nearest Power Pellet X" + "\nActivation Value: {:.6f}"
                        elif j == 11:
                            string = "Input Layer: Nearest Power Pellet Y" + "\nActivation Value: {:.6f}"
                        elif j == 12:
                            string = "Input Layer: Power Pellet Currently Active" + "\nActivation Value: {:.6f}"
                    elif i == 1:
                        string = "Hidden Layer 1: Node " + str(j) + "\nActivation Value: {:.6f}"
                    elif i == 2:
                        string = "Hidden Layer 2: Node " + str(j) + "\nActivation Value: {:.6f}"
                    elif i == 3:
                        string = "Hidden Layer 3: Node " + str(j) + "\nActivation Value: {:.6f}"
                    elif i == 4:
                        if j == 0:
                            string = "Output Layer: Up" + "\nActivation Value: {:.6f}"
                        elif j == 1:
                            string = "Output Layer: Down" + "\nActivation Value: {:.6f}"
                        elif j == 2:
                            string = "Output Layer: Left" + "\nActivation Value: {:.6f}"
                        elif j == 3:
                            string = "Output Layer: Right" + "\nActivation Value: {:.6f}"
                    self.hover_tracker.widget.setToolTip(string.format(
                        self.neural_network.layers[i].nodes[j].activation_value))

    def toggle_state(self, button):
        if button.text() == "Centered Start Position":
            if button.isChecked():
                self.agent_interface.set_game_start_pos({
                    'player_start': vec(13, 23),
                    'player_respawn': vec(13, 23),
                    'blinky': BLINKY_START_POS,
                    'inky': INKY_START_POS,
                    'pinky': PINKY_START_POS,
                    'clyde': CLYDE_START_POS
                })
            else:
                self.agent_interface.set_game_start_pos({
                    'player_start': PLAYER_START_POS,
                    'player_respawn': PLAYER_RESPAWN_POS,
                    'blinky': BLINKY_START_POS,
                    'inky': INKY_START_POS,
                    'pinky': PINKY_START_POS,
                    'clyde': CLYDE_START_POS
                })
        if button.text() == "Default Start Position":
            if button.isChecked():
                self.agent_interface.set_game_start_pos({
                    'player_start': PLAYER_START_POS,
                    'player_respawn': PLAYER_RESPAWN_POS,
                    'blinky': BLINKY_START_POS,
                    'inky': INKY_START_POS,
                    'pinky': PINKY_START_POS,
                    'clyde': CLYDE_START_POS
                })
            else:
                self.agent_interface.set_game_start_pos({
                    'player_start': vec(13, 23),
                    'player_respawn': vec(13, 23),
                    'blinky': BLINKY_START_POS,
                    'inky': INKY_START_POS,
                    'pinky': PINKY_START_POS,
                    'clyde': CLYDE_START_POS
                })

    def toggle_p_pellet_active(self):
        self.up_plot.show_p_active = not self.up_plot.show_p_active
        self.down_plot.show_p_active = not self.down_plot.show_p_active
        self.left_plot.show_p_active = not self.left_plot.show_p_active
        self.right_plot.show_p_active = not self.right_plot.show_p_active
        self.up_plot.update_plot()
        self.down_plot.update_plot()
        self.left_plot.update_plot()
        self.right_plot.update_plot()

    def toggle_p_pellet_inactive(self):
        self.up_plot.show_p_inactive = not self.up_plot.show_p_inactive
        self.down_plot.show_p_inactive = not self.down_plot.show_p_inactive
        self.left_plot.show_p_inactive = not self.left_plot.show_p_inactive
        self.right_plot.show_p_inactive = not self.right_plot.show_p_inactive
        self.up_plot.update_plot()
        self.down_plot.update_plot()
        self.left_plot.update_plot()
        self.right_plot.update_plot()

    def toggle_rand_decisions(self):
        self.up_plot.show_rand = not self.up_plot.show_rand
        self.down_plot.show_rand = not self.down_plot.show_rand
        self.left_plot.show_rand = not self.left_plot.show_rand
        self.right_plot.show_rand = not self.right_plot.show_rand
        self.up_plot.update_plot()
        self.down_plot.update_plot()
        self.left_plot.update_plot()
        self.right_plot.update_plot()

    def check_state(self, button):
        if button.text() == "Show Power Pellet Active":
            self.up_plot.show_p_active = not self.up_plot.show_p_active
            self.down_plot.show_p_active = not self.down_plot.show_p_active
            self.left_plot.show_p_active = not self.left_plot.show_p_active
            self.right_plot.show_p_active = not self.right_plot.show_p_active
        if button.text() == "Show Power Pellet Inactive":
            self.up_plot.show_p_inactive = not self.up_plot.show_p_inactive
            self.down_plot.show_p_inactive = not self.down_plot.show_p_inactive
            self.left_plot.show_p_inactive = not self.left_plot.show_p_inactive
            self.right_plot.show_p_inactive = not self.right_plot.show_p_inactive
        if button.text() == "Show Random Decisions":
            self.up_plot.show_rand = not self.up_plot.show_rand
            self.down_plot.show_rand = not self.down_plot.show_rand
            self.left_plot.show_rand = not self.left_plot.show_rand
            self.right_plot.show_rand = not self.right_plot.show_rand
        self.up_plot.update_plot()
        self.down_plot.update_plot()
        self.left_plot.update_plot()
        self.right_plot.update_plot()

    def load_screen(self):
        # Initialize layout
        self.center_widget = QWidget()
        layout = QVBoxLayout()
        self.tabs.setStyleSheet(QTAB_STYLE)
        self.tabs.addTab(self.main_tab_UI(), "Start")
        self.tabs.addTab(self.explainAI_tab_UI(), "Model Insights")
        #self.tabs.addTab(self.network_tab_UI(), "Network")
        #self.tabs.addTab(self.advanced_options_tab_UI(), "Options")
        self.tabs.addTab(self.about_tab_UI(), "About")
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
        left_layout.addSpacing(1)

        # Create dropdown menu
        infobox = CollapsibleBox(subtitle='Information')
        b_layout = QVBoxLayout()
        text = QLabel(main_info_text)
        text.setWordWrap(True)
        text.setStyleSheet(TEXT_STYLE)
        text.setAlignment(QtCore.Qt.AlignTop)
        text.setMinimumSize(500, 120)
        b_layout.addWidget(text)
        infobox.setContentLayout(b_layout)
        left_layout.addWidget(infobox)
        left_layout.addSpacing(10)

        # Create the Target High Score input box and button
        hlayout1 = QHBoxLayout()
        self.tar_high_score_input = QLineEdit(self.window)
        self.tar_high_score_input.setMinimumSize(300, 30)
        self.tar_high_score_input.setStyleSheet(QLINE_STYLE)
        self.tar_high_score_input.setText("400")
        self.tar_high_score_button = QPushButton('Set Target High Score', self.window)
        self.tar_high_score_button.clicked.connect(self.highScoreButton)
        self.tar_high_score_button.setMinimumSize(180, 30)
        self.tar_high_score_button.setStyleSheet(BUTTON_STYLE)
        self.tar_high_score_button.setToolTip(target_high_score_tooltip)
        hlayout1.addWidget(self.tar_high_score_input, 1)
        hlayout1.addSpacing(5)
        hlayout1.addWidget(self.tar_high_score_button)
        left_layout.addLayout(hlayout1)
        left_layout.addSpacing(10)

        # Create the Learning Rate slider and button
        hlayout2 = QHBoxLayout()
        self.learning_rate_slider = QSlider(Qt.Horizontal)
        self.learning_rate_slider.setMinimum(0)
        self.learning_rate_slider.setMaximum(10000)
        self.learning_rate_slider.setValue(500)
        self.learning_rate_slider.setTickInterval(5)
        self.learning_rate_slider.setMinimumSize(100, 30)
        self.learning_rate_slider.valueChanged.connect(self.slider_valuechange)
        self.learning_rate_button = QPushButton('Set Learning Rate')
        self.learning_rate_button.setMinimumSize(180, 30)
        self.learning_rate_button.clicked.connect(self.learningRateButton)
        self.learning_rate_button.setStyleSheet(BUTTON_STYLE)
        self.learning_rate_button.setToolTip(learning_rate_tooltip)
        hlayout2.addWidget(self.learning_rate_slider)
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

        # Create Options
        options_layout = QVBoxLayout()
        self.start_symm = QRadioButton('Centered Start Position')
        self.start_symm.toggled.connect(lambda: self.toggle_state(self.start_symm))
        self.start_symm.setToolTip(starting_locations_tooltip)
        self.start_symm.setStyleSheet(TEXT_STYLE)
        self.start_default = QRadioButton('Default Start Position')
        self.start_default.toggled.connect(lambda: self.toggle_state(self.start_default))
        self.start_default.setToolTip(starting_locations_tooltip)
        self.start_default.setStyleSheet(TEXT_STYLE)


        options_layout.addWidget(self.start_symm)
        options_layout.addWidget(self.start_default)
        left_layout.addLayout(options_layout)
        left_layout.addSpacing(10)

        # Create the Button for the Begin button to start the game (sim)
        self.begin_button = QPushButton('Start Learning Agent', self.window)
        self.begin_button.setStyleSheet(BUTTON_STYLE)
        self.begin_button.setMinimumSize(130, 30)
        self.begin_button.setToolTip(start_button_tooltip)
        left_layout.addWidget(self.begin_button)
        self.begin_button.clicked.connect(self.beginButton)
        left_layout.addSpacing(10)

        # Create the Stop button to terminate the simulation
        self.stop_button = QPushButton('Stop Learning Agent', self.window)
        self.stop_button.setStyleSheet(BUTTON_STYLE)
        self.stop_button.setMinimumSize(130, 30)
        left_layout.addWidget(self.stop_button)
        self.stop_button.clicked.connect(self.stopButton)
        left_layout.addSpacing(10)
        left_layout.addStretch()

        # Right Panel Title
        self.visualization_label = QLabel('Neural Network Activity', self.window)
        self.visualization_label.setStyleSheet(TITLE_STYLE)
        right_layout.addWidget(self.visualization_label)

        # Visualizer Layout
        vis_layout = QHBoxLayout()
        input_label_layout = QVBoxLayout()
        input_label_layout.addSpacing(25)
        output_label_layout = QVBoxLayout()
        output_label_layout.addSpacing(45)

        for i_label in self.input_labels:
            qlabel = QLabel(i_label)
            qlabel.setStyleSheet(TEXT_STYLE)
            input_label_layout.addWidget(qlabel)
            input_label_layout.addSpacing(2)

        for o_label in self.output_labels:
            qlabel = QLabel(o_label)
            qlabel.setStyleSheet(TEXT_STYLE)
            output_label_layout.addWidget(qlabel)
            output_label_layout.addSpacing(1)

        input_label_layout.addSpacing(10)
        output_label_layout.addSpacing(30)

        # Initialize the Visualizer
        self.visualizer = Visualizer(self.neural_network, self.diagram_width, self.diagram_height, self.agent_interface)
        vis_layout.addLayout(input_label_layout)
        vis_layout.addSpacing(2)
        vis_layout.addWidget(self.visualizer)
        vis_layout.addSpacing(2)
        vis_layout.addLayout(output_label_layout)
        vis_layout.addSpacing(6)
        right_layout.addLayout(vis_layout)
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
        main_layout = QHBoxLayout()
        plots_layout = QGridLayout()

        up_layout = QVBoxLayout()
        up_inner_layout = QHBoxLayout()
        self.up_label = QLabel("UP")
        self.up_label.setStyleSheet(TITLE_STYLE)
        up_layout.addWidget(self.up_label)
        up_layout.addLayout(up_inner_layout)
        up_inner_layout.addWidget(self.up_plot.plot_widget)

        down_layout = QVBoxLayout()
        down_inner_layout = QHBoxLayout()
        self.down_label = QLabel("DOWN")
        self.down_label.setStyleSheet(TITLE_STYLE)
        down_layout.addWidget(self.down_label)
        down_layout.addLayout(down_inner_layout)
        down_inner_layout.addWidget(self.down_plot.plot_widget)

        left_layout = QVBoxLayout()
        left_inner_layout = QHBoxLayout()
        self.left_label = QLabel("LEFT")
        self.left_label.setStyleSheet(TITLE_STYLE)
        left_layout.addWidget(self.left_label)
        left_layout.addLayout(left_inner_layout)
        left_inner_layout.addWidget(self.left_plot.plot_widget)

        right_layout = QVBoxLayout()
        right_inner_layout = QHBoxLayout()
        self.right_label = QLabel("RIGHT")
        self.right_label.setStyleSheet(TITLE_STYLE)
        right_layout.addWidget(self.right_label)
        right_layout.addLayout(right_inner_layout)
        right_inner_layout.addWidget(self.right_plot.plot_widget)

        side_panel_layout = QVBoxLayout()
        checkbox_layout = QVBoxLayout()
        self.show_p_pellet_active = QCheckBox("Show Power Pellet Active")
        self.show_p_pellet_active.setStyleSheet(TEXT_STYLE)
        self.show_p_pellet_active.toggled.connect(lambda: self.check_state(self.show_p_pellet_active))
        self.show_p_pellet_active.setToolTip(plot_active_tooltip)
        self.show_p_pellet_inactive = QCheckBox("Show Power Pellet Inactive")
        self.show_p_pellet_inactive.setStyleSheet(TEXT_STYLE)
        self.show_p_pellet_inactive.toggled.connect(lambda: self.check_state(self.show_p_pellet_inactive))
        self.show_p_pellet_inactive.setToolTip(plot_inactive_tooltip)
        self.show_p_pellet_inactive.click()
        self.show_rand_decisions = QCheckBox("Show Random Decisions")
        self.show_rand_decisions.setStyleSheet(TEXT_STYLE)
        self.show_rand_decisions.toggled.connect(lambda: self.check_state(self.show_rand_decisions))
        self.show_rand_decisions.setToolTip(plot_rand_tooltip)
        checkbox_layout.addWidget(self.show_p_pellet_active)
        checkbox_layout.addWidget(self.show_p_pellet_inactive)
        checkbox_layout.addWidget(self.show_rand_decisions)
        timer2_layout = QHBoxLayout()
        time_label = QLabel('Running time')
        time_label.setStyleSheet(TEXT_STYLE)
        timer2_layout.addWidget(time_label)
        self.timer2_label = QLabel('0:00')
        self.timer2_label.setStyleSheet(TEXT_STYLE)
        self.timer2_label.setAlignment(Qt.AlignRight)
        timer2_layout.addWidget(self.timer2_label)
        timer2_layout.addSpacing(5)
        side_panel_layout.addLayout(checkbox_layout)
        legend_layout = QHBoxLayout()
        legend_i_layout = QVBoxLayout()
        legend_widget = LegendWidget()
        legend_layout.addWidget(legend_widget)
        b_label = QLabel('Pac-Man')
        b_label.setStyleSheet(TEXT_STYLE)
        legend_i_layout.addWidget(b_label)
        b_label = QLabel('Blinky')
        b_label.setStyleSheet(TEXT_STYLE)
        legend_i_layout.addWidget(b_label)
        b_label = QLabel('Inky')
        b_label.setStyleSheet(TEXT_STYLE)
        legend_i_layout.addWidget(b_label)
        b_label = QLabel('Pinky')
        b_label.setStyleSheet(TEXT_STYLE)
        legend_i_layout.addWidget(b_label)
        b_label = QLabel('Clyde')
        b_label.setStyleSheet(TEXT_STYLE)
        legend_i_layout.addWidget(b_label)
        b_label = QLabel('Nearest Pellet')
        b_label.setStyleSheet(TEXT_STYLE)
        legend_i_layout.addWidget(b_label)
        b_label = QLabel('Nearest Power Pellet')
        b_label.setStyleSheet(TEXT_STYLE)
        legend_i_layout.addWidget(b_label)
        legend_layout.addLayout(legend_i_layout)
        legend_layout.addStretch()
        side_panel_layout.addLayout(legend_layout)

        infobox = CollapsibleBox()
        b_layout = QVBoxLayout()
        text = QLabel(plot_tab_text)
        text.setWordWrap(True)
        text.setStyleSheet(TEXT_STYLE)
        text.setAlignment(QtCore.Qt.AlignTop)
        text.setMinimumSize(200, 250)
        b_layout.addWidget(text)
        infobox.setContentLayout(b_layout)
        side_panel_layout.addWidget(infobox)
        side_panel_layout.addStretch()
        side_panel_layout.addLayout(timer2_layout)

        plots_layout.addLayout(up_layout, 0, 0)
        plots_layout.addLayout(down_layout, 0, 1)
        plots_layout.addLayout(left_layout, 1, 0)
        plots_layout.addLayout(right_layout, 1, 1)

        main_layout.addLayout(plots_layout)
        main_layout.addSpacing(20)
        main_layout.addLayout(side_panel_layout)
        main_layout.addSpacing(20)

        explainAI_tab_layout.addLayout(main_layout)
        explainAITab.setLayout(explainAI_tab_layout)
        return explainAITab

    """
    def network_tab_UI(self):
        networkTab = QWidget()
        network_tab_layout = QVBoxLayout()
        vis_layout = QHBoxLayout()
        input_label_layout = QVBoxLayout()
        input_label_layout.addSpacing(35)
        input_layout = QVBoxLayout()
        input_layout.addSpacing(35)
        output_label_layout = QVBoxLayout()
        output_label_layout.addSpacing(65)

        for i_label in self.input_labels:
            qlabel = QLabel(i_label)
            qlabel.setStyleSheet(TEXT_STYLE)
            input_label_layout.addWidget(qlabel)
            qlabel2 = QLabel('')
            qlabel2.setStyleSheet(TEXT_STYLE)
            input_layout.addWidget(qlabel2)
            self.input_qlabels.append(qlabel2)

        for o_label in self.output_labels:
            qlabel = QLabel(o_label)
            qlabel.setStyleSheet(TEXT_STYLE)
            output_label_layout.addWidget(qlabel)

        input_label_layout.addSpacing(115)
        input_layout.addSpacing(115)
        output_label_layout.addSpacing(140)
        self.tab_vis = Visualizer(self.neural_network, 600, 550, self.agent_interface, x_scale=2, y_scale=1.5)
        vis_layout.addLayout(input_label_layout)
        vis_layout.addSpacing(2)
        vis_layout.addLayout(input_layout)
        vis_layout.addSpacing(2)
        vis_layout.addWidget(self.tab_vis)
        vis_layout.addSpacing(2)
        vis_layout.addLayout(output_label_layout)
        vis_layout.addSpacing(6)
        self.tab_vis.setStyleSheet(WIDGET_STYLE)
        self.hover_tracker = HoverTracker(self.tab_vis)
        self.hover_tracker.positionChanged.connect(self.on_position_changed)
        network_tab_layout.addLayout(vis_layout)
        networkTab.setLayout(network_tab_layout)
        return networkTab

    def update_network_tab(self, ghosts, nearest_pellet, nearest_p_pellet, p_active, decision):
        self.input_qlabels[0].setText("{:.0f}".format(ghosts[0].x))
        self.input_qlabels[1].setText("{:.0f}".format(ghosts[0].y))
        self.input_qlabels[2].setText("{:.0f}".format(ghosts[1].x))
        self.input_qlabels[3].setText("{:.0f}".format(ghosts[1].y))
        self.input_qlabels[4].setText("{:.0f}".format(ghosts[2].x))
        self.input_qlabels[5].setText("{:.0f}".format(ghosts[2].y))
        self.input_qlabels[6].setText("{:.0f}".format(ghosts[3].x))
        self.input_qlabels[7].setText("{:.0f}".format(ghosts[3].y))
        self.input_qlabels[8].setText("{:.0f}".format(nearest_pellet[0]))
        self.input_qlabels[9].setText("{:.0f}".format(nearest_pellet[1]))
        self.input_qlabels[10].setText("{:.0f}".format(nearest_p_pellet[0]))
        self.input_qlabels[11].setText("{:.0f}".format(nearest_p_pellet[1]))
        self.input_qlabels[12].setText(str(p_active))
    """

    def advanced_options_tab_UI(self):
        advancedOptionsTab = QWidget()
        advanced_options_tab_layout = QVBoxLayout()
        advancedOptionsTab.setLayout(advanced_options_tab_layout)
        return advancedOptionsTab

    def about_tab_UI(self):
        aboutTab = QWidget()
        about_tab_layout = QVBoxLayout()

        about_label = QLabel(about_tab_text)
        about_label.setWordWrap(True)
        about_label.setStyleSheet(TEXT_STYLE)
        about_label.setAlignment(QtCore.Qt.AlignLeft)
        about_tab_layout.addSpacing(20)
        about_tab_layout.addWidget(about_label)

        infobox = CollapsibleBox()
        b_layout = QVBoxLayout()
        text = QLabel(credits_text)
        text.setWordWrap(True)
        text.setStyleSheet(TEXT_STYLE)
        text.setAlignment(QtCore.Qt.AlignTop)
        b_layout.addWidget(text)
        b_layout.addSpacing(2)
        github_link = HyperlinkLabel()
        github_link.setText('<a href={0}>{1}</a>'.format('https://github.com/mikeanthony321/PacMan_Learner', 'Project GitHub'))
        github_link.setStyleSheet(TEXT_STYLE)
        b_layout.addWidget(github_link)
        infobox.setContentLayout(b_layout)

        about_tab_layout.addWidget(infobox)
        about_tab_layout.addStretch()
        aboutTab.setLayout(about_tab_layout)
        return aboutTab

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

class LegendWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.pac_brush = pg.mkBrush(245, 210, 5, 255)
        self.pellet_brush = pg.mkBrush(220, 220, 220, 0)
        self.p_pellet_brush = pg.mkBrush(142, 240, 67, 0)
        self.blinky_brush = pg.mkBrush(255, 0, 0, 255)
        self.pinky_brush = pg.mkBrush(255, 184, 255, 255)
        self.inky_brush = pg.mkBrush(0, 255, 255, 255)
        self.clyde_brush = pg.mkBrush(255, 184, 82, 255)
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(self.pac_brush)
        painter.setPen(QPen(QColor(255, 225, 15), 1))
        painter.drawEllipse(2, 5, 10, 10)
        painter.setBrush(self.blinky_brush)
        painter.setPen(QPen(QColor(255, 0, 0, 255), 0))
        painter.drawEllipse(2, 30, 10, 10)
        painter.setBrush(self.inky_brush)
        painter.setPen(QPen(QColor(0, 255, 255), 0))
        painter.drawEllipse(2, 55, 10, 10)
        painter.setBrush(self.pinky_brush)
        painter.setPen(QPen(QColor(255, 184, 255), 0))
        painter.drawEllipse(2, 80, 10, 10)
        painter.setBrush(self.clyde_brush)
        painter.setPen(QPen(QColor(255, 184, 82), 0))
        painter.drawEllipse(2, 105, 10, 10)
        painter.setBrush(self.pellet_brush)
        painter.setPen(QPen(QColor(220, 220, 220), 0))
        painter.drawEllipse(2, 130, 10, 10)
        painter.setBrush(self.p_pellet_brush)
        painter.setPen(QPen(QColor(142, 240, 67), 2))
        painter.drawEllipse(2, 155, 10, 10)

    def minimumSizeHint(self):
        return QSize(20, 170)

class Visualizer(QWidget):
    def __init__(self, network_diagram, width, height, interface, x_scale=1, y_scale=1):
        super().__init__()
        # Initialize the diagram
        self.title = "Visualizer"
        self.width = width
        self.height = height
        self.node_size = int(self.height / 18)
        self.base_color = (51, 199, 255)
        self.color_val_param = 1
        self.thickness_param = 3
        self.base_line_thickness_param = 1
        self.x_scale = x_scale
        self.y_scale = y_scale

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
                        painter.drawLine(
                            (int(self.network.layers[i].nodes[j].x * self.x_scale) + math.floor((self.node_size / 2))),
                            (int(self.network.layers[i].nodes[j].y * self.y_scale) + math.floor((self.node_size / 2))),
                            (int(self.network.layers[i - 1].nodes[k].x * self.x_scale) + math.floor(
                                (self.node_size / 2))),
                            (int(self.network.layers[i - 1].nodes[k].y * self.y_scale) + math.floor(
                                (self.node_size / 2))))

        # Draw nodes
        painter.setPen(QPen(self.transformColor(-0.5, 1), 1))
        for a in range(len(self.network.layers)):
            for b in range(len(self.network.layers[a].nodes)):
                radialGradient = QRadialGradient(
                    QPoint((int(self.network.layers[a].nodes[b].x * self.x_scale) + math.floor(self.node_size / 2)),
                           int(self.network.layers[a].nodes[b].y * self.y_scale) + math.floor(self.node_size / 2)), 40)

                node_color_1 = self.transformColor(self.network.layers[a].nodes[b].get_activation_value(), 0.5)
                node_color_2 = self.transformColor(self.network.layers[a].nodes[b].get_activation_value(), 0.75)
                node_color_3 = self.transformColor(self.network.layers[a].nodes[b].get_activation_value(), 1.25)

                radialGradient.setColorAt(0.1, node_color_1)
                radialGradient.setColorAt(0.5, node_color_2)
                radialGradient.setColorAt(1.0, node_color_3)
                painter.setBrush(QBrush(radialGradient))
                painter.drawEllipse(int(self.network.layers[a].nodes[b].x * self.x_scale),
                                    int(self.network.layers[a].nodes[b].y * self.y_scale), self.node_size,
                                    self.node_size)

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
        return QSize(self.width, self.height)

    def sizeHint(self):
        return QSize(self.width, self.height)

    def update_diagram(self, network_diagram):
        self.network = network_diagram
        self.repaint()


class BorderWidget(QFrame):

    def __init__(self, *args):
        super(BorderWidget, self).__init__(*args)
        self.setStyleSheet(WIDGET_STYLE)

    def minimumSizeHint(self):
        return QSize(550, 100)


class PlotStruct:

    def __init__(self):
        self.show_rand = False
        self.show_p_inactive = False
        self.show_p_active = False
        self.avg_display = True
        self.current_display = False
        self.point_size = 10
        self.base_opacity = 90
        self.plot_limit = 120
        self.decisions_window = 14

        self.pac_spot = {
            'pos': [0, 0],
            'pen': {'color': (245, 210, 5, 255), 'width': 0},
            'size': 18,
            'brush': pg.mkBrush(235, 200, 5, 255)
        }
        self.pellet_brush = pg.mkBrush(220, 220, 220, self.base_opacity)
        self.p_pellet_brush = pg.mkBrush(142, 240, 67, self.base_opacity)
        self.blinky_brush = pg.mkBrush(255, 0, 0, self.base_opacity)
        self.pinky_brush = pg.mkBrush(255, 184, 255, self.base_opacity)
        self.inky_brush = pg.mkBrush(0, 255, 255, self.base_opacity)
        self.clyde_brush = pg.mkBrush(255, 184, 82, self.base_opacity)

        self.inactive_exploitation_points = []
        self.inactive_exploration_points = []
        self.active_exploitation_points = []
        self.active_exploration_points = []

        self.inactive_exploitation_avg = []
        self.inactive_exploration_avg = []
        self.active_exploitation_avg = []
        self.active_exploration_avg = []

        self.inactive_exploitation_avg_points = []
        self.inactive_exploration_avg_points = []
        self.active_exploitation_avg_points = []
        self.active_exploration_avg_points = []

        self.inactive_exploitation_decisions_counter = 0
        self.inactive_exploration_decisions_counter = 0
        self.active_exploitation_decisions_counter = 0
        self.active_exploration_decisions_counter = 0

        self.plot_item = pg.ScatterPlotItem()
        self.plot_item.addPoints([self.pac_spot])
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setXRange(-30, 30)
        self.plot_widget.setYRange(-30, 30)
        self.plot_widget.addItem(self.plot_item)
        self.plot_ref = None

    def update_points(self, ghosts, nearest_pellet, nearest_p_pellet, p_active, decision_type):
        blinky_spot = {
            'pos': [ghosts[0].x, ghosts[0].y],
            'size': self.point_size,
            'pen': {'color': (255, 0, 0, self.base_opacity), 'width': 0},
            'brush': self.blinky_brush
        }
        inky_spot = {
            'pos': [ghosts[1].x, ghosts[1].y],
            'size': self.point_size,
            'pen': {'color': (0, 255, 255, self.base_opacity), 'width': 0},
            'brush': self.inky_brush
        }
        pinky_spot = {
            'pos': [ghosts[2].x, ghosts[2].y],
            'size': self.point_size,
            'pen': {'color': (255, 184, 255, self.base_opacity), 'width': 0},
            'brush': self.pinky_brush
        }
        clyde_spot = {
            'pos': [ghosts[3].x, ghosts[3].y],
            'size': self.point_size,
            'pen': {'color': (255, 184, 82, self.base_opacity), 'width': 0},
            'brush': self.clyde_brush
        }
        pellet_spot = {
            'pos': [nearest_pellet[0], nearest_pellet[1]],
            'size': self.point_size,
            'pen': {'color': (220, 220, 220, self.base_opacity), 'width': 0},
            'brush': self.pellet_brush
        }
        p_pellet_spot = {
            'pos': [nearest_p_pellet[0], nearest_p_pellet[1]],
            'size': self.point_size,
            'pen': {'color': (142, 240, 67, self.base_opacity), 'width': 0},
            'brush': self.p_pellet_brush
        }

        if p_active:
            if decision_type == "EXPLORATION":
                self.active_exploration_points.extend(
                    [blinky_spot, pinky_spot, inky_spot, clyde_spot, pellet_spot, p_pellet_spot])
                self.active_exploration_avg = self.update_avg(
                    self.active_exploration_avg, ghosts, nearest_pellet, nearest_p_pellet,
                    self.active_exploration_decisions_counter)
                self.active_exploration_decisions_counter += 1
                if self.active_exploration_decisions_counter == self.decisions_window:
                    self.active_exploration_avg_points = self.add_avg_point(
                        self.active_exploration_avg, self.active_exploration_avg_points)
                    self.active_exploration_decisions_counter = 0
                    self.active_exploration_avg = []
                if len(self.active_exploration_points) >= self.plot_limit:
                    self.active_exploration_points = self.active_exploration_points[6:]
            else:
                self.active_exploitation_points.extend(
                    [blinky_spot, pinky_spot, inky_spot, clyde_spot, pellet_spot, p_pellet_spot])
                self.active_exploitation_avg = self.update_avg(
                    self.active_exploitation_avg, ghosts, nearest_pellet, nearest_p_pellet,
                    self.active_exploitation_decisions_counter)
                self.active_exploitation_decisions_counter += 1
                if self.active_exploitation_decisions_counter == self.decisions_window:
                    self.active_exploitation_avg_points = self.add_avg_point(
                        self.active_exploitation_avg, self.active_exploitation_avg_points)
                    self.active_exploitation_decisions_counter = 0
                    self.active_exploitation_avg = []
                if len(self.active_exploitation_points) >= self.plot_limit:
                    self.active_exploitation_points = self.active_exploitation_points[6:]
        else:
            if decision_type == "EXPLORATION":
                self.inactive_exploration_points.extend(
                    [blinky_spot, pinky_spot, inky_spot, clyde_spot, pellet_spot, p_pellet_spot])
                self.inactive_exploration_avg = self.update_avg(
                    self.inactive_exploration_avg, ghosts, nearest_pellet, nearest_p_pellet,
                    self.inactive_exploration_decisions_counter)
                self.inactive_exploration_decisions_counter += 1
                if self.inactive_exploration_decisions_counter == self.decisions_window:
                    self.inactive_exploration_avg_points = self.add_avg_point(
                        self.inactive_exploration_avg, self.inactive_exploration_avg_points)
                    self.inactive_exploration_decisions_counter = 0
                    self.inactive_exploration_avg = []
                if len(self.inactive_exploration_points) >= self.plot_limit:
                    self.inactive_exploration_points = self.inactive_exploration_points[6:]
            else:
                self.inactive_exploitation_points.extend(
                    [blinky_spot, pinky_spot, inky_spot, clyde_spot, pellet_spot, p_pellet_spot])
                self.inactive_exploitation_avg = self.update_avg(
                    self.inactive_exploitation_avg, ghosts, nearest_pellet, nearest_p_pellet,
                    self.inactive_exploitation_decisions_counter)
                self.inactive_exploitation_decisions_counter += 1
                if self.inactive_exploitation_decisions_counter == self.decisions_window:
                    self.inactive_exploitation_avg_points = self.add_avg_point(
                        self.inactive_exploitation_avg, self.inactive_exploitation_avg_points)
                    self.inactive_exploitation_decisions_counter = 0
                    self.inactive_exploitation_avg = []
                if len(self.inactive_exploitation_points) >= self.plot_limit:
                    self.inactive_exploitation_points = self.inactive_exploitation_points[6:]


    def update_avg(self, avg_list, ghosts, nearest_pellet, nearest_p_pellet, counter):
        if counter == 0:
            avg_list.append(ghosts[0].x)
            avg_list.append(ghosts[0].y)
            avg_list.append(ghosts[1].x)
            avg_list.append(ghosts[1].y)
            avg_list.append(ghosts[2].x)
            avg_list.append(ghosts[2].y)
            avg_list.append(ghosts[3].x)
            avg_list.append(ghosts[3].y)
            avg_list.append(nearest_pellet[0])
            avg_list.append(nearest_pellet[1])
            avg_list.append(nearest_p_pellet[0])
            avg_list.append(nearest_p_pellet[1])
        elif 0 < counter <= self.decisions_window:
            avg_list[0] = (avg_list[0] * ((counter - 1)/ counter)) + \
                          (ghosts[0].x * (1 / counter))
            avg_list[1] = (avg_list[1] * ((counter - 1) / counter)) + \
                          (ghosts[0].y * (1 / counter))
            avg_list[2] = (avg_list[2] * ((counter - 1) / counter)) + \
                          (ghosts[1].x * (1 / counter))
            avg_list[3] = (avg_list[3] * ((counter - 1) / counter)) + \
                          (ghosts[1].y * (1 / counter))
            avg_list[4] = (avg_list[4] * ((counter - 1) / counter)) + \
                          (ghosts[2].x * (1 / counter))
            avg_list[5] = (avg_list[5] * ((counter - 1) / counter)) + \
                          (ghosts[2].y * (1 / counter))
            avg_list[6] = (avg_list[6] * ((counter - 1) / counter)) + \
                          (ghosts[3].x * (1 / counter))
            avg_list[7] = (avg_list[7] * ((counter - 1) / counter)) + \
                          (ghosts[3].y * (1 / counter))
            avg_list[8] = (avg_list[8] * ((counter - 1) / counter)) + \
                          (nearest_pellet[0] * (1 / counter))
            avg_list[9] = (avg_list[9] * ((counter - 1) / counter)) + \
                          (nearest_pellet[1] * (1 / counter))
            avg_list[10] = (avg_list[10] * ((counter - 1) / counter)) + \
                           (nearest_p_pellet[0] * (1 / counter))
            avg_list[11] = (avg_list[11] * ((counter - 1) / counter)) + \
                           (nearest_p_pellet[1] * (1 / counter))

        return avg_list

    def add_avg_point(self, avg_list, point_list):
        blinky_spot = {
            'pos': [avg_list[0], avg_list[1]],
            'size': self.point_size,
            'pen': {'color': (255, 0, 0, self.base_opacity), 'width': 0},
            'brush': self.blinky_brush
        }
        inky_spot = {
            'pos': [avg_list[2], avg_list[3]],
            'size': self.point_size,
            'pen': {'color': (0, 255, 255, self.base_opacity), 'width': 0},
            'brush': self.inky_brush
        }
        pinky_spot = {
            'pos': [avg_list[4], avg_list[5]],
            'size': self.point_size,
            'pen': {'color': (255, 184, 255, self.base_opacity), 'width': 0},
            'brush': self.pinky_brush
        }
        clyde_spot = {
            'pos': [avg_list[6], avg_list[7]],
            'size': self.point_size,
            'pen': {'color': (255, 184, 82, self.base_opacity), 'width': 0},
            'brush': self.clyde_brush
        }
        pellet_spot = {
            'pos': [avg_list[8], avg_list[9]],
            'size': self.point_size,
            'pen': {'color': (220, 220, 220, self.base_opacity), 'width': 0},
            'brush': self.pellet_brush
        }
        p_pellet_spot = {
            'pos': [avg_list[10], avg_list[11]],
            'size': self.point_size,
            'pen': {'color': (142, 240, 67, self.base_opacity), 'width': 0},
            'brush': self.p_pellet_brush
        }
        point_list.extend(
            [blinky_spot, pinky_spot, inky_spot, clyde_spot, pellet_spot, p_pellet_spot])

        return point_list

    def update_plot(self):
        plot_points = []
        if self.avg_display:
            if self.show_p_active and self.show_p_inactive and self.show_rand:
                plot_points.extend(self.active_exploitation_avg_points)
                plot_points.extend(self.inactive_exploitation_avg_points)
                plot_points.extend(self.inactive_exploration_points)
                plot_points.extend(self.active_exploration_points)
            elif self.show_p_active and self.show_p_inactive and not self.show_rand:
                plot_points.extend(self.active_exploitation_avg_points)
                plot_points.extend(self.inactive_exploitation_avg_points)
            elif self.show_p_active and not self.show_p_inactive and self.show_rand:
                plot_points.extend(self.active_exploitation_avg_points)
                plot_points.extend(self.active_exploration_points)
            elif not self.show_p_active and self.show_p_inactive and self.show_rand:
                plot_points.extend(self.inactive_exploitation_avg_points)
                plot_points.extend(self.inactive_exploration_points)
            elif self.show_p_active and not self.show_p_inactive and not self.show_rand:
                plot_points.extend(self.active_exploitation_avg_points)
            elif not self.show_p_active and self.show_p_inactive and not self.show_rand:
                plot_points.extend(self.inactive_exploitation_avg_points)
            else:
                plot_points = []
        elif self.current_display:
            if self.show_p_active and self.show_p_inactive and self.show_rand:
                plot_points.extend(self.active_exploitation_avg_points)
                plot_points.extend(self.inactive_exploitation_avg_points)
                plot_points.extend(self.inactive_exploration_points)
                plot_points.extend(self.active_exploration_points)
            elif self.show_p_active and self.show_p_inactive and not self.show_rand:
                plot_points.extend(self.active_exploitation_avg_points)
                plot_points.extend(self.inactive_exploitation_avg_points)
            elif self.show_p_active and not self.show_p_inactive and self.show_rand:
                plot_points.extend(self.active_exploitation_avg_points)
                plot_points.extend(self.active_exploration_points)
            elif not self.show_p_active and self.show_p_inactive and self.show_rand:
                plot_points.extend(self.inactive_exploitation_avg_points)
                plot_points.extend(self.inactive_exploration_points)
            elif self.show_p_active and not self.show_p_inactive and not self.show_rand:
                plot_points.extend(self.active_exploitation_avg_points)
            elif not self.show_p_active and self.show_p_inactive and not self.show_rand:
                plot_points.extend(self.inactive_exploitation_avg_points)
            else:
                plot_points = []

        self.plot_item.setData(plot_points)
        self.plot_widget.addItem(self.plot_item)




class CollapsibleBox(QWidget):
    def __init__(self, title=None, subtitle=None, parent=None):
        super(CollapsibleBox, self).__init__(parent)

        self.hlayout = QHBoxLayout()

        if title is not None:
            self.title_label = QLabel(title)
            self.title_label.setStyleSheet(TITLE_STYLE)
            self.hlayout.addWidget(self.title_label)
            self.hlayout.addSpacing(0)
        elif subtitle is not None:
            self.title_label = QLabel(subtitle)
            self.title_label.setStyleSheet(TEXT_STYLE)
            self.hlayout.addWidget(self.title_label)
            self.hlayout.addSpacing(0)

        self.toggle_button = QToolButton(checkable=True, checked=False)
        self.toggle_button.setStyleSheet(TEXT_STYLE)
        self.toggle_button.setStyleSheet("QToolButton { border: none; }")
        self.toggle_button.setToolButtonStyle(
            Qt.ToolButtonTextBesideIcon
        )
        self.toggle_button.setArrowType(Qt.DownArrow)
        self.toggle_button.pressed.connect(self.on_pressed)

        self.toggle_animation = QtCore.QParallelAnimationGroup(self)

        self.content_area = QScrollArea(maximumHeight=0, minimumHeight=0)
        self.content_area.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed
        )
        self.content_area.setFrameShape(QFrame.NoFrame)

        lay = QVBoxLayout(self)
        lay.setSpacing(0)
        lay.setContentsMargins(0, 0, 0, 0)
        self.hlayout.addWidget(self.toggle_button)
        self.hlayout.addStretch(2)
        lay.addLayout(self.hlayout)
        lay.addWidget(self.content_area)

        self.toggle_animation.addAnimation(
            QPropertyAnimation(self, b"minimumHeight")
        )
        self.toggle_animation.addAnimation(
            QPropertyAnimation(self, b"maximumHeight")
        )
        self.toggle_animation.addAnimation(
            QPropertyAnimation(self.content_area, b"maximumHeight")
        )

    @QtCore.pyqtSlot()
    def on_pressed(self):
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(
            Qt.UpArrow if not checked else Qt.DownArrow
        )
        self.toggle_animation.setDirection(
            QAbstractAnimation.Forward
            if not checked
            else QAbstractAnimation.Backward
        )
        self.toggle_animation.start()

    def setContentLayout(self, layout):
        lay = self.content_area.layout()
        del lay
        self.content_area.setLayout(layout)
        collapsed_height = (
                self.sizeHint().height() - self.content_area.maximumHeight()
        )
        content_height = layout.sizeHint().height()
        for i in range(self.toggle_animation.animationCount()):
            animation = self.toggle_animation.animationAt(i)
            animation.setDuration(500)
            animation.setStartValue(collapsed_height)
            animation.setEndValue(collapsed_height + content_height)

        content_animation = self.toggle_animation.animationAt(
            self.toggle_animation.animationCount() - 1
        )
        content_animation.setDuration(500)
        content_animation.setStartValue(0)
        content_animation.setEndValue(content_height)


class HoverTracker(QtCore.QObject):
    positionChanged = QtCore.pyqtSignal(QPoint)

    def __init__(self, widget):
        super().__init__(widget)
        self._widget = widget
        self.widget.setMouseTracking(True)
        self.widget.installEventFilter(self)

    @property
    def widget(self):
        return self._widget

    def eventFilter(self, obj, event):
        if obj is self.widget and event.type() == QtCore.QEvent.MouseMove:
            self.positionChanged.emit(event.pos())
        return super().eventFilter(obj, event)

class HyperlinkLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__()
        self.setStyleSheet('font-size: 35px')
        self.setOpenExternalLinks(True)
        self.setParent(parent)