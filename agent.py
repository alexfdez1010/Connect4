from abc import ABCMeta, abstractmethod
from random import choice
from numpy import array


# Interface for the agent, that it is going to be the opponent of the AI
class Agent(metaclass=ABCMeta):

    # @id_player : id corresponding to the agent
    # @num_col : numbers of col of the environment
    def __init__(self, id_player: int, num_actions: int):
        self.__id_player = id_player
        self.__num_actions = num_actions

    # The agent ought to implement this function to perform the action depending of the state
    @abstractmethod
    def action(self, state: array) -> int:
        pass

    # Setter of the id
    # @id_player : new id for the agent
    def set_id_player(self, id_player):
        self.__id_player = id_player

    # Getter of the id
    def get_id_player(self):
        return self.__id_player

    def get_num_actions(self):
        return self.__num_actions


# This agent allow the user to play against the AI
class Human(Agent):

    def __init__(self, id_player: int, num_actions: int):
        super(Human, self).__init__(id_player, num_actions)

    # Allow the user to select an action
    def action(self, state: array) -> int:

        print("The valid columns are: ", end="")
        for i in range(self.get_num_actions()):
            if state[i] == 0:
                print(f"{i + 1}", end=" ")

        action = int(input()) - 1

        while action < 0 or action >= self.get_num_actions() or state[action] != 0:
            print("You must input a valid column")
            action = int(input()) - 1

        return action


# This agent allow the AI to play against other AI, also against itself
class AI(Agent):

    # @id_player : id corresponding to the agent
    # @num_col : numbers of col of the environment
    # @model : model that its going to decide the actions, must implement the stable-baselines interface
    def __init__(self, id_player: int, num_actions: int, model):
        super(AI, self).__init__(id_player, num_actions)
        self.__model = model

    # Getter of the model
    def get_model(self):
        return self.__model

    # Setter of the model
    # @model : new model for the agent
    def set_model(self, model):
        self.__model = model

    # Take the action based in the state
    def action(self, state: array) -> int:

        action = self.__model.predict(state)[0]

        counter = 0  # We use a counter to assure to not produce a infinite loop

        while state[action] != 0 and counter > 100:
            action = self.__model.predict(state)
            counter += 1

        # In case the counter is higher than 100 implies that the
        # model cannot select a valid action so we select a random
        # action between the allowed actions
        if counter > 100:
            action = choice([i for i in range(self.get_num_actions()) if state[i] != 0])

        return action
