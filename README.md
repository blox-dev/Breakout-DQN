# Breakout-DQN
This project uses the Tensorflow framework to implement a Deep Q-Learning Network for the popular Atari game, Breakout.
#### Model details:
- 2 convolutional layers (32 filters, 4 strides and 64 filters, 2 strides)
- 1 dense layer with 512 neurons
- 1 dense output layer, 4 actions
- Adam optimizer, 5e-4 learning rate
- Huber loss function

This is the best model I have tried, but the convergence is still slow.
It reaches some level of intelligent play (or mean reward 4) after ~300.000 frames.
Best observed mean reward: 18 after ~4.000.000 frames.

#### Usage:
- run ```trainer.py``` for training a new model or continue training
- run ```player.py``` for testing the trained models

#### Further improvement:
- Limit frames per episode, some games take way longer to play than others.
- Better model