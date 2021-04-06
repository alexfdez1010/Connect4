from stable_baselines3.a2c import A2C
from stable_baselines3.a2c.policies import ActorCriticPolicy

from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import MlpPolicy

from environment import Connect4, NUM_COL
from agent import AI

from os import listdir
from sys import argv

NAME_SAVE_DQN: str = "DQN_Connect4"
NAME_SAVE_A2C: str = "A2C_Connect4"


def train(num_epochs: int = 250, num_steps: int = 1000, use_heuristic: bool = False):
    env_dqn = Connect4(None, use_heuristic=use_heuristic)
    env_a2c = Connect4(None, use_heuristic=use_heuristic)

    if NAME_SAVE_DQN + ".zip" in listdir():
        model_dqn = DQN.load(NAME_SAVE_DQN + ".zip", env_dqn)
    else:
        model_dqn = DQN(policy=MlpPolicy, env=env_dqn)

    if NAME_SAVE_A2C + ".zip" in listdir():
        model_a2c = A2C.load(NAME_SAVE_A2C + ".zip", env_a2c)
    else:
        model_a2c = A2C(policy=ActorCriticPolicy, env=env_a2c)

    agent_dqn = AI(2, NUM_COL, model_dqn)
    agent_a2c = AI(2, NUM_COL, model_a2c)

    env_dqn.change_agent(agent_a2c)
    env_a2c.change_agent(agent_dqn)

    for i in range(num_epochs):
        print(f"Executing epoch {i + 1}/{num_epochs}")
        model_dqn.learn(num_steps)
        model_dqn.save(NAME_SAVE_DQN)
        model_a2c.learn(num_steps)
        model_a2c.save(NAME_SAVE_A2C)


if __name__ == '__main__':

    if len(argv) > 1:
        epochs = int(argv[1])
        steps = int(argv[2])
        train(epochs, steps, use_heuristic=True)
    else:
        train(use_heuristic=True)
