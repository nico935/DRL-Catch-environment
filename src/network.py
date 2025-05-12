import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dims: Tuple[int, int, int], #  (84, 84, 4)
        n_actions: int                   
    ):   
        super(NeuralNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions

        self.conv1 = nn.Conv2d(input_dims[2], 32, kernel_size=8, stride=4) 

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Flatten the output of conv3 to feed into fully connected layers
        # Calculate the flattened size: 64 * 7 * 7
        self.fc1_dims = 64 * 7 * 7
        self.fc1 = nn.Linear(self.fc1_dims, 512)
        self.fc2 = nn.Linear(512, self.n_actions)
        """!
        Initialize a neural network. This network can be used to approximate
        some functions, maybe the reward function? Keep in mind that this class
        should only be used to define the neural network, not to train it using
        reinforcement learning algorithms.
        """


    def forward(
        self, state: np.ndarray
    ) -> torch.Tensor:
        """!
        Convert the input state to the relevant output using the neural network.
        """
        x = state.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x)) # why do we apply relu here? and not in the conv
        x = F.relu(self.conv3(x))

        # Flatten the output for the fully connected layer
        x = x.reshape(-1, self.fc1_dims) # or x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        q_values = self.fc2(x) # Raw Q-values

        return q_values
