from settings import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QLineEdit, QPushButton, QLabel, QTableWidget, QTableWidgetItem

class Analytics(QMainWindow):
    def __init__(self, monitor_size):
        super().__init__()
        self.window = QMainWindow()
        self.window.resize(WIDTH, HEIGHT)
        #PyQt5 vs PyGame do not position from the same spot and I couldn't for the life of me find what the exact
        #height of the top bar of the PyQt5 window is by default and had to mess around with values to find 31. Problem
        #is that I can't guarantee this isn't different on different OS, so this is more a temp measure.
        self.window.move(monitor_size.width() / 2 - WIDTH, monitor_size.height() / 2 - HEIGHT / 2 - 31)
        self.window.setWindowTitle('Pac-Man Learner Analytics')

        self.running = False

# -- -- -- GENERAL FUNCTIONS -- -- -- #
    def buttonClicked(self):
        self.running = not self.running
        print(self.running)

    def updateScreen(self):
        if not self.running:
            self.start_screen()
        else:
            self.sim_screen()


# -- -- -- START MENU FUNCTIONS -- -- -- #
    def start_screen(self):
        self.tar_high_score_label = QLabel('Target High Score', self.window)
        self.tar_high_score_label.move(20, 35)
        self.tar_high_score_input = QLineEdit(self.window)
        self.tar_high_score_input.move(20, 60)
        self.tar_high_score_input.resize(150, 30)
        self.tar_high_score_button = QPushButton('Submit', self.window)
        self.tar_high_score_button.move(170, 60)

        self.learning_rate_label = QLabel('Learning Rate', self.window)
        self.learning_rate_label.move(20, 105)
        self.learning_rate_input = QLineEdit(self.window)
        self.learning_rate_input.move(20, 130)
        self.learning_rate_input.resize(150, 30)
        self.learning_rate_button = QPushButton('Submit', self.window)
        self.learning_rate_button.move(170, 130)

        self.begin_label = QLabel('Begin Sim', self.window)
        self.begin_label.move(170, 250)
        self.begin_button = QPushButton('Begin', self.window)
        self.begin_button.move(170, 275)
        self.begin_button.clicked.connect(self.buttonClicked)

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