from random import randint

import numpy as np
from gym import Env
from gym.spaces import Discrete, Box

from agent import Human

NUM_COL: int = 7  # Constant to determinate the number of columns
NUM_ROW: int = 6  # Constant to determinate the number of rows

# Constants to describe the rewards that it going to get from the environment
LOSING_REWARD: int = -2
DRAWING_REWARD: int = 0
WINNING_REWARD: int = 2
HEURISTIC_REWARD: float = 0.2


# An environment to play Connect4 using Env gym interface
class Connect4(Env):

    # @agent : the opponent of the AI, must implement the interface Agent of this repository
    # @use_heuristic : a boolean indicating whatever use an heuristic for the rewards or not
    # The action space will correspond to a discrete number indicating in
    # which column is going to be insert the piece
    # The observation space will be a flatten array
    def __init__(self, agent, use_heuristic=False):

        self.agent = agent
        self.use_heuristic = use_heuristic

        self.action_space = Discrete(NUM_COL)
        self.observation_space = Box(low=0, high=2, shape=(NUM_COL * NUM_ROW + 1,), dtype=np.uint8)

        self.id_player = None
        self.state = None

    # Take a step in the environment and also the action taken by the agent
    # Afterwards it will return a tuple (observation,reward,done,info)
    # In case the agent is human it will render the board after
    # the action has been performed

    # Observation is the state of the game after taking the actions,
    # is a flatten array containing the value of the box where 0 is empty,
    # 1 occupied by first player and 2 by the second player

    # Reward will contain
    # WINNING_REWARD in case of winning,
    # DRAW_REWARD in case of drawing and
    # LOSE_REWARD in case of losing
    # If use_heuristic is true it will
    # return a value distinct of zero
    # in no-terminal states

    # done if true it will indicate that the episode has finished
    # otherwise false

    # info contains extra information, in this case it
    # will return the identifications of the players

    # @action : action to perform, its value must be in the range [0,NUM_COL-1]
    def step(self, action: int) -> tuple:
        reward = 0
        done = False
        info = {"id_player": self.id_player, "id_agent": self.agent.get_id_player()}
        if self.__valid_move(action):
            self.__update_state(action, self.id_player)
            if isinstance(self.agent, Human):
                self.render()
            if self.__is_finish(self.id_player):
                reward = WINNING_REWARD
                done = True
            elif self.__is_draw():
                reward = DRAWING_REWARD
                done = True
            else:
                new_action = self.agent.action(self.state)
                self.__update_state(new_action, self.agent.get_id_player())
                if self.__is_finish(self.agent.get_id_player()):
                    reward = LOSING_REWARD
                    done = True
                elif self.__is_draw():
                    done = True
                elif self.use_heuristic:
                    reward = self.__heuristic()
        else:
            reward = -1

        return self.state, reward, done, info

    # Reset the environment and if the agent has the id 1
    # it will allow it to perform its first action
    # In case the agent is human, it will show the pieces
    # corresponding to the human
    # It will return the first observation/state
    # This function must be called before using step in a new episode
    def reset(self) -> np.array:
        self.state = np.zeros(NUM_ROW * NUM_COL + 1)
        self.id_player = randint(1, 2)

        self.state[-1] = self.id_player
        self.agent.set_id_player((self.id_player & 1) + 1)

        if isinstance(self.agent, Human):

            if self.agent.get_id_player() == 1:
                print("You have the O pieces")
            else:
                print("You have the X pieces")

        if self.id_player == 2:
            action = self.agent.action(self.state)
            self.__update_state(action, self.agent.get_id_player())

        return self.state

    # Render the environment allowing to the user to watch the state of the board
    # @mode : if mode == 'human' it will render otherwise not
    def render(self, mode='human'):

        if mode == 'human':

            for _ in range(NUM_COL):
                print("----", end="")
            print("-\n", end="")

            for i in range(NUM_ROW):
                for j in range(NUM_COL):
                    if self.state[i * NUM_COL + j] == 1:
                        char = "O"
                    elif self.state[i * NUM_COL + j] == 2:
                        char = "X"
                    else:
                        char = " "
                    print(f"| {char} ", end="")
                    if j == NUM_COL - 1:
                        print("|", end="")
                print("\n", end="")
                for _ in range(NUM_COL):
                    print("----", end="")
                print("-\n", end="")
            print("\n", end="")

    # Change the agent performing the actions
    # @agent : new agent
    def change_agent(self, agent):
        self.agent = agent

    def __update_state(self, action: int, id_player: int) -> None:

        j = (NUM_ROW - 1) * NUM_COL + action

        while j >= 0 and self.state[j] != 0:
            j -= NUM_COL

        self.state[j] = id_player
        self.state[-1] = (id_player & 1) + 1

    def __valid_move(self, action):
        if self.state[action] != 0:
            return False
        else:
            return True

    def __is_finish(self, id_player) -> bool:

        for i in range(NUM_ROW):
            for j in range(NUM_COL):

                if self.state[i * NUM_COL + j] == id_player:

                    count = 0
                    r, c = i + 1, j

                    while r < NUM_ROW and self.state[r * NUM_COL + c] == id_player:
                        count += 1
                        r += 1

                    if count >= 3:
                        return True

                    count = 0
                    r, c = i + 1, j + 1

                    while r < NUM_ROW and c < NUM_COL and self.state[r * NUM_COL + c] == id_player:
                        count += 1
                        r += 1
                        c += 1

                    if count >= 3:
                        return True

                    count = 0
                    r, c = i, j + 1

                    while c < NUM_COL and self.state[r * NUM_COL + c] == id_player:
                        count += 1
                        c += 1

                    if count >= 3:
                        return True

        return False

    def __is_draw(self):

        return all(self.state != 0)

    def __heuristic(self) -> float:

        return self.__calculate_heuristic(self.id_player) \
               - 2 * self.__calculate_heuristic(self.agent.get_id_player())

    def __calculate_heuristic(self, id_player: int) -> float:

        est_value = 0

        for i in range(NUM_ROW):
            for j in range(NUM_COL):

                if self.state[i * NUM_COL + j] == id_player:

                    count = 0
                    r, c = i + 1, j

                    while r < NUM_ROW and self.state[r * NUM_COL + c] == id_player:
                        count += 1
                        r += 1

                    if count == 3:
                        est_value += HEURISTIC_REWARD

                    count = 0
                    r, c = i + 1, j + 1

                    while r < NUM_ROW and c < NUM_COL and self.state[r * NUM_COL + c] == id_player:
                        count += 1
                        r += 1
                        c += 1

                    if count == 3:
                        est_value += HEURISTIC_REWARD

                    count = 0
                    r, c = i, j + 1

                    while c < NUM_COL and self.state[r * NUM_COL + c] == id_player:
                        count += 1
                        c += 1

                    if count == 3:
                        est_value += HEURISTIC_REWARD

        return est_value
