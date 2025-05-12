import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class NeuralNetwork(nn.Module):
    def __init__(
        self,
        input_dims: Tuple[int, int, int], # (height, width, channels) e.g. (84, 84, 4)
        n_actions: int                    # e.g. 3
    ):   
        super(NeuralNetwork, self).__init__()
        self.input_dims = input_dims # Should be (channels, height, width) for PyTorch
        self.n_actions = n_actions

        # PyTorch expects channels first: (C, H, W)
        # The environment provides (H, W, C)
        # We will handle the permutation in the agent or forward pass

        # Example CNN architecture (adjust as needed)
        # Input: (Batch_size, 4, 84, 84)
        self.conv1 = nn.Conv2d(input_dims[2], 32, kernel_size=8, stride=4) # input_dims[2] is number of channels (fps) 84
        # Output conv1: ((84-8)/4 + 1) = 20 -> (Batch_size, 32, 20, 20)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # Output conv2: ((20-4)/2 + 1) = 9 -> (Batch_size, 64, 9, 9)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Output conv3: ((9-3)/1 + 1) = 7 -> (Batch_size, 64, 7, 7)

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
