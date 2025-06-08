
# Value-based RL Algorithms in Catch Environment

This repository implements several value-based reinforcement learning (RL) algorithms to solve the Catch game, a simple grid-world environment. 

## Algorithms Implemented

The following value-based RL algorithms are implemented:

- **Deep Q-Learning (DQN)** (Mnih et al., 2015)
- **Double Deep Q-Learning** (Hasselt et al.,2016)
- **Dueling Network Architectures**  (Wang et al., 2016)
- **Deep Quality-Value** (Sabatelli et al.,2020)
- **Deep Quality-Value Max** Learning (Sabatelli et al., 2020)

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

## References

Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A. A., Veness, J., Bellemare, M. G., ... & Hassabis, D. (2015). Human-level control through deep reinforcement learning. nature, 518(7540), 529-533.

Van Hasselt, H., Guez, A., & Silver, D. (2016, March). Deep reinforcement learning with double q-learning. In Proceedings of the AAAI conference on artificial intelligence (Vol. 30, No. 1).

Wang, Z., Schaul, T., Hessel, M., Hasselt, H., Lanctot, M., & Freitas, N. (2016, June). Dueling network architectures for deep reinforcement learning. In International conference on machine learning (pp. 1995-2003). PMLR.

Sabatelli, M., Louppe, G., Geurts, P., & Wiering, M. A. (2020, July). The deep quality-value family of deep reinforcement learning algorithms. In 2020 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.
