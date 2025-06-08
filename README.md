
# Value-based RL Algorithms in Catch Environment

This repository implements several value-based reinforcement learning (RL) algorithms to solve the classic Catch game, a simple grid-world environment. 

## Algorithms Implemented

The following value-based RL algorithms are implemented:

- **DQN (Deep Q-Network)**
  - Standard Q-learning with a neural network function approximator and target network.
- **DDQN (Double Deep Q-Network)**
  - Addresses Q-value overestimation in DQN by decoupling action selection and evaluation.
- **DQV (Deep QV-Learning)**
  - Learns both Q-value and Value functions simultaneously for improved stability.
- **DQV-Max**
  - A variant of DQV that further stabilizes value estimation using a max operator.

Each agent is implemented as a class in `src/agent.py`, inheriting from a common abstract base.

## The Catch Environment

The Catch game is implemented in `src/environment.py` as a subclass of OpenAI Gym’s `Env`. The environment features:

- A 21x21 grid world.
- Paddle controlled by the agent at the bottom; ball falls from the top.
- The agent must catch the falling ball with the paddle to receive rewards.
- Discrete action space: move left, move right, or stay.
- The observation state consists of the last 4 frames augmented  (shape: 84x84x4).

## Project Structure

- `src/agent.py` - RL agent classes (DQN, DDQN, DQV, DQV-Max).
- `src/environment.py` - Catch environment.
- `src/network.py` - Neural network for Q and Value functions.
- `src/main.py` - Training and evaluation scripts.
- `src/plotting.py` - Plotting utilities for results and training curves.
- `src/configs/` - Configuration files for experiments.

## Getting Started

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run training:**
   ```bash
   python src/main.py
   ```
