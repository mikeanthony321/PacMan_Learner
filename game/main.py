import os
import sys
from PyQt5.QtWidgets import QApplication

from game.pacman import Pacman
from game.agent import LearnerAgent
from game.settings import WIDTH, HEIGHT
from game.analytics_frame import Analytics
from game.analytics_test import testAgent

def main():
    app = QApplication(sys.argv)
    monitor = app.primaryScreen()
    monitor_size = monitor.size()
    os.environ['SDL_VIDEO_WINDOW_POS'] = '%d,%d' % (monitor_size.width() / 2 + WIDTH / 2, monitor_size.height() / 2 - HEIGHT / 2)
    game = Pacman(monitor_size)

    LearnerAgent.create_agent_instance(game)

    test_agent = testAgent()
    Analytics.create_analytics_instance(monitor_size, test_agent)

    game.run()


    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
