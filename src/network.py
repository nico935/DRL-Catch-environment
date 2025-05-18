import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class QNetwork(nn.Module): #QNetwork rename
    def __init__(
        self,
        input_dims: Tuple[int, int, int], #  (84, 84, 4)
        n_actions: int                   
    ):   
        super(QNetwork, self).__init__()
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
        x = F.relu(self.conv2(x)) 
        x = F.relu(self.conv3(x))

        # Flatten the output for the fully connected layer
        x = x.reshape(-1, self.fc1_dims) 

        x = F.relu(self.fc1(x))
        q_values = self.fc2(x) 

        return q_values

class SmallerNeuralNetwork(nn.Module): #smaller QNetwork rename
    def __init__(
        self,
        input_dims: Tuple[int, int, int], # (H, W, C) e.g., (84, 84, 4)
        n_actions: int
    ):
        super(SmallerNeuralNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions

        in_channels = input_dims[2] 


        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        # Input 84x84 -> Output (84-8)/4+1 = 20x20. Shape: (batch, 16, 20, 20)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # Input 20x20 -> Output (20-4)/2+1 = 9x9. Shape: (batch, 32, 9, 9)

        # Calculate the flattened size after conv layers
        self.fc1_dims = 32 * 9 * 9 # 32 filters * 9 height * 9 width = 2592

        self.fc1 = nn.Linear(self.fc1_dims, 128) # Reduced from 512 to 128
        self.fc2 = nn.Linear(128, self.n_actions)

    def forward(self, state: torch.Tensor) -> torch.Tensor: 

        x = state.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values



class DuelingNeuralNetwork(nn.Module):
    def __init__(self, input_dims: Tuple[int, int, int], n_actions: int):
        super(DuelingNeuralNetwork, self).__init__()
        self.input_dims = input_dims 
        self.n_actions = n_actions
        in_channels = self.input_dims[2]

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # Input 20x20 -> Output ((20-4)/2)+1 = 9x9. Shape: (batch, 32, 9, 9)

        self.conv_output_flat_size = 32 * 9 * 9  # 2592

        # Value Stream
        # Takes flattened conv output, outputs a single value V(s)
        self.value_fc1 = nn.Linear(self.conv_output_flat_size, 128) # Hidden layer for value stream
        self.value_output = nn.Linear(128, 1)                  # Output V(s)

        # Advantage Stream
        # Takes flattened conv output, outputs n_actions advantages A(s,a)
        self.advantage_fc1 = nn.Linear(self.conv_output_flat_size, 128) # Hidden layer for advantage stream
        self.advantage_output = nn.Linear(128, n_actions)           # Output A(s,a) for each action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state.permute(0, 3, 1, 2)

        # Pass through convolutional base
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the output from conv base
        x = torch.flatten(x, start_dim=1)

        # Value Stream
        value = F.relu(self.value_fc1(x))
        value = self.value_output(value)  # Shape: (batch_size, 1)

        # Advantage Stream
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_output(advantage) # Shape: (batch_size, n_actions)

        # Combine Value and Advantage streams
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # 'advantage.mean(dim=1, keepdim=True)' calculates mean along the action dimension
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

class VNetwork(nn.Module):
    def __init__(self, input_dims: Tuple[int, int, int]): # (H, W, C)
        super(VNetwork, self).__init__()
        self.input_dims = input_dims 

        in_channels = self.input_dims[2]

        
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        # Input 20x20 -> Output ((20-4)/2)+1 = 9x9. Shape: (batch, 32, 9, 9)
        self.conv_output_flat_size = 32 * 9 * 9  # 2592
        
        self.fc_value = nn.Linear(self.conv_output_flat_size, 128) 
        self.value_output = nn.Linear(128, 1) 

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = state.permute(0, 3, 1, 2) 
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten the output from conv
        x = torch.flatten(x, start_dim=1)

        value = F.relu(self.fc_value(x))
        value = self.value_output(value)  

        return value