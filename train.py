from stable_baselines3.a2c import A2C
from stable_baselines3.a2c.policies import ActorCriticPolicy

from stable_baselines3.dqn import DQN
from stable_baselines3.dqn.policies import MlpPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from environment import Connect4, NUM_COL
from agent import AI
from gym import spaces

from os import listdir
from sys import argv

import torch.nn as nn
from torch import no_grad, as_tensor, Tensor

NAME_SAVE_DQN: str = "DQN_Connect4"
NAME_SAVE_A2C: str = "A2C_Connect4"


class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 256, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        )

        with no_grad():
            n_flatten = self.cnn(
                as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim),
                                    nn.ReLU()
                                    )

    def forward(self, observations: Tensor):
        return self.linear(self.cnn(observations))


policy_kwargs = dict(
    feature_extractor_class=CustomCNN,
    feature_extractor_kwargs=dict(feature_dim=128)
)


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
        train(epochs, steps, use_heuristic=False)
    else:
        train(use_heuristic=False)
