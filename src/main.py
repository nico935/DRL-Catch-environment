from environment import CatchEnv
from agent import DQNAgent 
import numpy as np
import torch
N_EPISODES = 10000

def run_environment():
    """!
    Run the environment with a random agent.
    Make sure to adapt this to train your own implementation.
    """
    env = CatchEnv()
    # state_dimensions: (height, width, channels) as per environment.py's obs_shape
    # obs_shape from env is (H, W, C) which is (84, 84, 4)
    state_dimensions = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(
        memory_size=1000000,       # Adjust as needed
        state_dimensions=state_dimensions,
        n_actions=n_actions,
        learning_rate=0.000025,    # Tunable
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.99901,     # Results in epsilon ~0.01 after ~1000 learn steps if learn called each step
        batch_size=32,
        target_update_frequency=10000 # Or based on episodes/steps
    )

    print(f"Using deviiice: {agent.device}")
    print(f"State dimensionnns: {state_dimensions}")
    print(f"Number of actions: {n_actions}")


    scores = [] # For logging average scores
    eps_history = []
    
    for ep in range(N_EPISODES):
        observation, info = env.reset() # observation is (H, W, C) numpy array
        terminated = False
        truncated = False # Gymnasium uses terminated and truncated
        score = 0
        # Choosing a random action instead of an informed one.
        # This is where you come in!
        #action = env.action_space.sample()
        # You probably want to use these values to train your agent :)
        while not terminated and not truncated:
            action = agent.choose_action(observation)
            new_observation, reward, terminated, truncated, info = env.step(action)

            agent.store_transition(observation, action, reward, new_observation, terminated or truncated)
            agent.learn()

            observation = new_observation
            score += reward

        scores.append(score)
        eps_history.append(agent.epsilon)
        avg_score = np.mean(scores[-100:]) # Moving average of last 100 scores

        print(f"episode {ep} | score: {score:.2f} | moving average reward: {avg_score:.2f} | epsilon: {agent.epsilon:.2f} | steps: {agent.mem_counter}")



if __name__ == "__main__":
    run_environment()