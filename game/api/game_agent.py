from abc import ABC, abstractmethod


class GameAgentAPI(ABC):

    # Sets app_state to 'game'
    @abstractmethod
    def gameStart(self):
        pass

    # Sets target high score
    def setTarHighScore(self, score):
        pass

    # Provide the list of actions the player is able to take, in the form of Action enums
    @abstractmethod
    def getAvailableActions(self):
        pass

    # Change the player's direction to move upwards
    @abstractmethod
    def moveUp(self):
        pass

    # Change the player's direction to move downwards
    @abstractmethod
    def moveDown(self):
        pass

    # Change the player's direction to move to the left
    @abstractmethod
    def moveLeft(self):
        pass

    # Change the player's direction to move to the right
    @abstractmethod
    def moveRight(self):
        pass

    # Provide the coordinates of the ghosts, in terms of grid-index offset from player
    # Provide as a vector, or some similar data structure
    @abstractmethod
    def getGhostsGridCoords(self):
        pass

    # Provide the coordinates of the pellet that is closest to the player, in terms of grid-index offset from player
    # Provide as a vector, or some similar data structure
    @abstractmethod
    def getNearestPelletGridCoords(self):
        pass

    # Provide the coordinates of the power pellet that is closest to the player, in terms of grid-index offset from player
    # Provide as a vector, or some similar data structure
    @abstractmethod
    def getNearestPowerPelletGridCoords(self):
        pass

    # Provide a boolean flag to indicate whether a power pellet is currently in effect
    @abstractmethod
    def isPowerPelletActive(self):
        pass

    # Sets app_state to 'game'
    @abstractmethod
    def gameStart(self):
        pass

    # sets positions of pac-man and ghosts
    @abstractmethod
    def set_start_pos(self, pos_dict):
        pass

    # Sets target high score
    def setTarHighScore(self, score):
        pass

    @abstractmethod
    def getReward(self):
        pass
