from settings import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit, QPushButton, QLabel, QTableWidget, QTableWidgetItem
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont

class Analytics(QMainWindow):
    def __init__(self, monitor_size):
        super().__init__()
        self.window = QMainWindow()
        self.window.resize(WIDTH, HEIGHT)
        self.window.move(monitor_size.width() / 2 - WIDTH, monitor_size.height() / 2 - HEIGHT / 2 - 31)
        self.window.setWindowTitle('Pac-Man Learner Analytics')

        self.running = False
        self.tar_high_score = 0
        self.learning_rate = 0.0

        self.timer_min = 0
        self.timer_sec = 0
        self.timer_ms = 0

        self.start_screen()

# -- -- -- GENERAL FUNCTIONS -- -- -- #
    def updateScreen(self):
        if not self.running:
            self.start_screen()
        else:
            self.sim_screen()

    def showTime(self):
        if self.running:
            self.timer_sec += 1
            if self.timer_sec >= 60:
                self.timer_min += 1
                self.timer_sec = 0

            self.formatTime()

    def formatTime(self):
        if self.timer_sec < 10:
            self.timer_label.setText("Execution timer: %d:0%d" % (self.timer_min, self.timer_sec))
        else:
            self.timer_label.setText("Execution timer: %d:%d" % (self.timer_min, self.timer_sec))

# -- -- -- BUTTON FUNCTIONS -- -- -- #
    def buttonClicked(self):
        self.timer.start(1000)
        self.sim_screen()
        self.running = True

    def highScoreButton(self):
        if not self.running:
            self.tar_high_score = self.tar_high_score_input.text()
            self.tar_high_score = int(self.tar_high_score)
            print("The target high score is: " + self.tar_high_score_input.text())
            self.tar_high_score_input.setText("")
        else:
            print("You must stop the sim to enter a target high score")
            self.tar_high_score_input.setText("")

    def learningRateButton(self):
        if not self.running:
            self.learning_rate = self.learning_rate_input.text()
            self.learning_rate = float(self.learning_rate)
            if self.learning_rate < 1.0:
                print(self.learning_rate)
            else: #I do not know enough about learning rate to know if we want to require it be under 1, just did to have a test
                print('Please enter a number less than 1')
            self.learning_rate_input.setText("")
        else:
            print("You must stop the sim to enter a new learning rate")
            self.learning_rate_input.setText("")


# -- -- -- START MENU FUNCTIONS -- -- -- #
    def start_screen(self):
        #Create the Label/Text Input/Button for the Target High Score
        self.tar_high_score_label = QLabel('Target High Score', self.window)
        self.tar_high_score_label.move(20, 35)
        self.tar_high_score_input = QLineEdit(self.window)
        self.tar_high_score_input.move(20, 60)
        self.tar_high_score_input.resize(150, 30)
        self.tar_high_score_button = QPushButton('Submit', self.window)
        self.tar_high_score_button.move(170, 60)

        #Create the Label/Text Input/Button for the Learning Rate
        self.learning_rate_label = QLabel('Learning Rate', self.window)
        self.learning_rate_label.move(20, 105)
        self.learning_rate_input = QLineEdit(self.window)
        self.learning_rate_input.move(20, 130)
        self.learning_rate_input.resize(150, 30)
        self.learning_rate_button = QPushButton('Submit', self.window)
        self.learning_rate_button.move(170, 130)

        #Create the Label/Button for the Begin button to start the game (sim)
        self.begin_label = QLabel('Begin Sim', self.window)
        self.begin_label.move(170, 250)
        self.begin_button = QPushButton('Begin', self.window)
        self.begin_button.move(170, 275)
        self.begin_button.clicked.connect(self.buttonClicked)

        #Create the Label and Timer for the Execution Timer
        #This would actually be present on the sim screen, not the main screen but for now it'll live on the main screen
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.showTime)
        self.timer_label = QLabel('Execution timer: 0:00', self.window)
        self.timer_label.move(380, 35)
        self.timer_label.resize(150,50)
        self.timer_label.setFont(QFont('Arial', 12))

        self.tar_high_score_button.clicked.connect(self.highScoreButton)
        self.learning_rate_button.clicked.connect(self.learningRateButton)
        self.window.show()



# -- -- -- SIM RUNNING FUNCTIONS -- -- -- #
    def sim_screen(self):
        self.q_value_table_label = QLabel('Table of Q Values', self.window)
        self.q_value_table_label.move(20, 35)
        self.q_value_table = QTableWidget(4, 4, self.window)
        self.q_value_table.setMinimumSize(500,200)
        self.createTable()
        self.q_value_table.move(20, 60)
        self.q_value_table.size()

        self.window.show()

    def createTable(self):
        self.q_value_table.setItem(0, 0, QTableWidgetItem("Cell (1,1)"))
        self.q_value_table.setItem(0, 1, QTableWidgetItem("Cell (1,2)"))
        self.q_value_table.setItem(1, 0, QTableWidgetItem("Cell (2,1)"))
        self.q_value_table.setItem(1, 1, QTableWidgetItem("Cell (2,2)"))
        self.q_value_table.setItem(2, 0, QTableWidgetItem("Cell (3,1)"))
        self.q_value_table.setItem(2, 1, QTableWidgetItem("Cell (3,2)"))
        self.q_value_table.setItem(3, 0, QTableWidgetItem("Cell (4,1)"))
        self.q_value_table.setItem(3, 1, QTableWidgetItem("Cell (4,2)"))