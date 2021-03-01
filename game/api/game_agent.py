from abc import ABC, abstractmethod

class GameAgentAPI(ABC):

    # The AI will use this method to constantly listen (on a separate thread) for the player to enter a new grid square
    # If the AI notices that a new grid square was entered, it will trigger the neural network to decide its move
    # If the AI notices that the player just died, it will backpropagate to improve its next attempt
    #
    # Maintain an int value in the game class that updates EVERY SINGLE FRAME
    # 0  -> Nothing happened this frame
    # 1  -> Player entered a new grid square this frame
    # 2  -> Player died this frame
    # -1 -> Simulation ended (either by user, or by reaching target score)
    @abstractmethod
    def getUpdateState(self):
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
