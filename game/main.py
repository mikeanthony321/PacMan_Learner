import os
import sys
from PyQt5.QtWidgets import QApplication

from pacman import Pacman
from agent import LearnerAgent
from settings import WIDTH, HEIGHT

from analytics_frame import Analytics

def main():
    app = QApplication(sys.argv)
    monitor = app.primaryScreen()
    monitor_size = monitor.size()
    os.environ['SDL_VIDEO_WINDOW_POS'] = '%d,%d' % (monitor_size.width() / 2 + WIDTH / 2, monitor_size.height() / 2 - HEIGHT / 2)

    game = Pacman(monitor_size)
    LearnerAgent.create_agent_instance(game)
    Analytics.create_analytics_instance(monitor_size, LearnerAgent.agent_instance)

    game.run()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
