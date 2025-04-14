from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import Tuple

class Agent():
    def __init__(
        self,
        memory_size: int,
        state_dimensions: Tuple[int, int, int],
        n_actions: int,
        # Add any other arguments you need here
        # e.g. learning rate, discount factor, etc.
    ) -> None:
        """!
        Initializes the agent.
        Agent is an abstract class that should be inherited by any agent that
        wants to interact with the environment. The agent should be able to
        store transitions, choose actions based on observations, and learn from the
        transitions.

        @param memory_size (int): Size of the memory buffer
        @param state_dimensions (int): Number of dimensions of the state space
        @param n_actions (int): Number of actions the agent can take
        """

        self.memory_size = memory_size
        self.state_buffer = np.zeros((self.memory_size, *state_dimensions), dtype=np.float32)
        self.new_state_buffer = np.zeros((self.memory_size, *state_dimensions), dtype=np.float32)
        self.action_buffer = np.zeros((self.memory_size, n_actions), dtype=np.int32)
        self.reward_buffer = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_buffer = np.zeros(self.memory_size, dtype=bool)

    def store_transition(
        self,
        state: np.ndarray,
        action: int, # Is this always an int?
        reward: float,
        new_state: np.ndarray,
        done: bool
    ) -> None:
        """!
        Stores the state transition for later memory replay.
        Make sure that the memory buffer does not exceed its maximum size.

        Hint: after reaching the limit of the memory buffer, maybe you should start overwriting
        the oldest transitions?

        @param state        (list): Vector describing current state
        @param action       (int): Action taken
        @param reward       (float): Received reward
        @param new_state    (list): Newly observed state.
        """

        pass

    @abstractmethod
    def choose_action(
        self,
        observation: np.ndarray
    ) -> int: # Is this always an int?
        """!
        Abstract method that should be implemented by the child class, e.g. DQN or DDQN agents.
        This method should contain the full logic needed to choose an action based on the current state.
        Maybe you can store the neural network in the agent class and use it here to decide which action to take?

        @param observation (np.ndarray): Vector describing current state

        @return (int): Action to take
        """

        return 0

    @abstractmethod
    def learn(self) -> None:
        """!
        Update the parameters of the internal networks.
        This method should be implemented by the child class.
        """

        pass
