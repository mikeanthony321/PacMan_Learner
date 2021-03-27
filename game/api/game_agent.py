from abc import ABC, abstractmethod

class GameAgentAPI(ABC):

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

    # Provide the coordinates of the player, in terms of grid indices
    # Provide as a vector, or some similar data structure
    @abstractmethod
    def getPlayerGridCoords(self):
        pass

    # Provide the coordinates of the ghost that is closest to the player, in terms of grid indices
    # Provide as a vector, or some similar data structure
    @abstractmethod
    def getNearestGhostGridCoords(self):
        pass

    # Provide the coordinates of the pellet that is closest to the player, in terms of grid indices
    # Provide as a vector, or some similar data structure
    @abstractmethod
    def getNearestPelletGridCoords(self):
        pass

    # Provide the coordinates of the power pellet that is closest to the player, in terms of grid indices
    # Provide as a vector, or some similar data structure
    @abstractmethod
    def getNearestPowerPelletGridCoords(self):
        pass

    # Provide a boolean flag to indicate whether a power pellet is currently in effect
    @abstractmethod
    def isPowerPelletActive(self):
        pass
