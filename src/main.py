from environment import CatchEnv
from agent import DQNAgent, DDQNAgent
import numpy as np
import torch
import matplotlib.pyplot as plt

N_EPISODES = 3000
DDQN=True
DQN=False
SEEDS = [42, 123, 789, 101, 555]
all_runs_scores = []
all_runs_moving_averages = []

def run_environment(seed_value):  
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    env = CatchEnv()
    # state_dimensions: (height, width, channels) as per environment.py's obs_shape
    # obs_shape from env is (H, W, C) which is (84, 84, 4)
    state_dimensions = env.observation_space.shape
    n_actions = env.action_space.n
    if DQN==True:
        agent = DQNAgent(
            memory_size=10000,       # Adjust as needed
            state_dimensions=state_dimensions,
            n_actions=n_actions,
            learning_rate=0.001,    # Tunable
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_min=0.0001,
            epsilon_decay=0.99801,     # Results in epsilon ~0.01 after ~1000 learn steps if learn called each step
            batch_size=32,
            target_update_frequency=200 
        )
    if DDQN==True:
        agent = DDQNAgent(
            memory_size=10000,       # Adjust as needed
            state_dimensions=state_dimensions,
            n_actions=n_actions,
            learning_rate=0.00025,    # Tunable
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_min=0.0001,
            epsilon_decay=0.999901,     # Results in epsilon ~0.01 after ~1000 learn steps if learn called each step
            batch_size=32,
            t_weight_start=0.005,
            t_weight_min=0.00001,
            t_weight_decay=0.999,
            target_update_frequency=2000,
            soft_update=False,
        )
    print(f"DQN {DQN}, DDQN {DDQN}")
    print(f"learning rate: {agent.lr}")
    print(f"Using deviiice: {agent.device}")
    print(f"State dimensionnns: {state_dimensions}")
    print(f"Number of actions: {n_actions}")
    print(f"--- Running Training for Seed: {seed_value} ---")



    eps_history = []
    current_run_score=[]
    current_run_moving_average=[]
    
    for ep in range(N_EPISODES):
        if ep == 0:
            observation = env.reset(seed=seed_value)
        else:
            observation = env.reset()
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

        current_run_score.append(score)
        avg_score=np.mean(current_run_score) #total average reward
        eps_history.append(agent.epsilon)
        if ep>= 100-1:
            run_avg_score = np.mean(current_run_score[-100:]) # Moving average of last 100 scores
        else:
            run_avg_score = np.mean(current_run_score)
        current_run_moving_average.append(run_avg_score)
        if (ep + 1) % 50 == 0:
            print(f"episode {ep} | score: {score:.2f} | mov avg reward: {run_avg_score:.2f} | avg reward {avg_score:.2f} epsilon: {agent.epsilon:.2f} | steps: {agent.mem_counter}")
    return current_run_score, current_run_moving_average

if __name__ == "__main__":
    for seed in SEEDS:
        raw_scores, moving_avg_scores = run_environment(seed)
        all_runs_scores.append(raw_scores)
        all_runs_moving_averages.append(moving_avg_scores)

    # --- Process and Plot Results ---
    # min_len_ma = min(len(ma) for ma in all_runs_moving_averages)
    # processed_ma_scores = np.array([ma[:min_len_ma] for ma in all_runs_moving_averages])

    mean_ma_scores = np.mean(all_runs_moving_averages, axis=0)
    std_ma_scores = np.std(all_runs_moving_averages, axis=0)
    # For a 95% confidence interval for the mean, you might use Standard Error:
    # from scipy.stats import sem
    # sem_ma_scores = sem(processed_ma_scores, axis=0)
    # conf_interval_upper = mean_ma_scores + 1.96 * sem_ma_scores
    # conf_interval_lower = mean_ma_scores - 1.96 * sem_ma_scores
    # Or simply use 1 or 2 standard deviations for a variance band

    episodes_axis = np.arange(1, N_EPISODES + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_axis, mean_ma_scores, label='Mean Moving Average Reward (across seeds)')

    # Shaded region for +/- 1 Standard Deviation
    plt.fill_between(episodes_axis,
                     mean_ma_scores - std_ma_scores,
                     mean_ma_scores + std_ma_scores,
                     alpha=0.2, label='Mean +/- 1 Std Dev')

    # Shaded region for +/- 2 Standard Deviations (optional, can make plot busy)
    # plt.fill_between(episodes_axis,
    #                  mean_ma_scores - 2 * std_ma_scores,
    #                  mean_ma_scores + 2 * std_ma_scores,
    #                  alpha=0.1, label='Mean +/- 2 Std Dev')

    plt.title(f'DDQN Performance on Catch Game (Avg over {len(SEEDS)} seeds)')
    plt.xlabel('Episode')
    plt.ylabel('100-Episode Moving Average Reward')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1.05) # Rewards are 0 or 1
    # plt.savefig('dqn_catch_performance.png') # Save the plot
    plt.show()

