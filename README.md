# Connect4

This repository provides an environment called Connect4 implemented using gym.Env interface as its name indicates is about the popular game Connect 4
This environment can be played by a human

train.py trains two AIs with models A2C and DQN, the concrete implementation is based in using two environments where we train each AI in a different environment and the opponent will be the other AI
main.py after training you can watch a playing by the two AIs or play against one of them

# Requirements

pip install stable-baselines3(also you can use stable-baselines changing the imports with stable-baselines3 to stable-baselines)
pip install gym(should be installed with the previous command)
