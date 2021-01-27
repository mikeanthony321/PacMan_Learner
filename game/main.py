import os
from pacman import *

def main():
    app = QApplication(sys.argv)
    monitor = app.primaryScreen()
    monitor_size = monitor.size()
    os.environ['SDL_VIDEO_WINDOW_POS'] = '%d,%d' % (monitor_size.width() / 2, monitor_size.height() / 2 - HEIGHT / 2)
    game = Pacman(monitor_size)
    game.run()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
