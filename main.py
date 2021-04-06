from stable_baselines3.a2c import A2C
from stable_baselines3.dqn import DQN

from environment import Connect4, WINNING_REWARD, DRAWING_REWARD, NUM_COL
from agent import Human, AI
from train import NAME_SAVE_DQN, NAME_SAVE_A2C, train

from os import listdir


def play(human_play=True, use_dqn_model=True):
    if human_play:

        env = Connect4(Human(2, NUM_COL))
        if use_dqn_model:
            if NAME_SAVE_DQN + ".zip" not in listdir():
                train(150, 500)
            model = DQN.load(NAME_SAVE_DQN + ".zip", env)
        else:
            if NAME_SAVE_A2C + ".zip" not in listdir():
                train(150, 500)
            model = A2C.load(NAME_SAVE_A2C + ".zip", env)

        obv = env.reset()

        done = False

        while not done:
            action = model.predict(obv)[0]

            print(f"The IA has put a piece in the column {action + 1}")

            obv, reward, done, info = env.step(action)

        env.render()
        if reward == WINNING_REWARD:
            print("Sorry, you have lost")

        elif reward == DRAWING_REWARD:
            print("Has been a tough battle, congratulations for you draw")

        else:
            print("Congratulations, you have won")

    else:

        if NAME_SAVE_DQN + ".zip" not in listdir() or NAME_SAVE_A2C + ".zip" not in listdir():
            train(150, 500)

        env = Connect4(None)

        agent_dqn = AI(2, NUM_COL, model=DQN.load(NAME_SAVE_DQN + ".zip", env=env))
        env.change_agent(agent_dqn)

        model_a2c = A2C.load(NAME_SAVE_A2C + ".zip", env)

        obv = env.reset()

        done = False

        while not done:
            env.render()
            action = model_a2c.predict(obv)[0]

            obv, reward, done, info = env.step(action)

        env.render()

        if reward == WINNING_REWARD:
            print("The winner is A2C")

        elif reward == DRAWING_REWARD:
            print("The result has been a draw")

        else:
            print("The winner is DQN")


if __name__ == '__main__':

    sel = 1

    while sel != 0:

        sel = int(input("Do you want to play against one of the AI(1) "
                        "or watch a playing between the two AIs(2). "
                        "If you want to quit input a 0: "))

        while not (0 <= sel <= 2):
            sel = int(input("Input a valid value: "))

        if sel == 1:

            sel = int(input("Do you want to play against A2C(1) or DQN(2): "))

            while not (1 <= sel <= 2):
                sel = int(input("Input a valid value: "))

            if sel == 1:
                play(human_play=True, use_dqn_model=False)
            else:
                play(human_play=True, use_dqn_model=True)

        elif sel == 2:

            play(False)
