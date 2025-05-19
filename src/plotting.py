#Plotting script used to visualize the results of different DQN algorithms in fig 1 and 2

import os
import json
import numpy as np
import matplotlib.pyplot as plt

base_drive_path = "/content/drive/MyDrive/Courses Groning/Deep Reinforcement Learning/Assignent 1/"
#local google drive path
experiment_files = {
    "DQN (SmallQNetwork)": os.path.join(base_drive_path, "DQN_Small_network_Experiment1/results_DQN_SmallQNetwork.json"),
    "DDQN (SmallQNetwork)": os.path.join(base_drive_path, "DDQN_Experiment1/results_DDQN_SmallQNetwork.json"),
    "Duelling DDQN": os.path.join(base_drive_path, "Duelling_DDQN_Experiment1/results_DDQN_DuelingNeuralNetwork.json"),
    "DQV": os.path.join(base_drive_path, "DQV_network_Experiment1/results_DQV_QNetwork_VNetwork.json"),
    "DQVMax": os.path.join(base_drive_path, "DQVMax_Experiment1/results_DQVMax_QNetwork_VNetwork.json"),
}
#epsilon min calculated from the results of the DQN experiment setup, DQV is slightly different by margin of 10
epsilon_min_episode = 3491

# --- Function to create a plot ---
def create_comparison_plot(metric_key, std_metric_key, title, ylabel, epsilon_episode_marker, save_filename=a):
    plt.figure(figsize=(15, 10))
    plot_handles = []

    for display_name, file_path in experiment_files.items():
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        mean_scores = np.array(data[metric_key])
        std_scores = np.array(data[std_metric_key])
        episodes = np.arange(1, len(mean_scores) + 1)

        line, = plt.plot(episodes, mean_scores, label=display_name) 
        plot_handles.append(line)
        plt.fill_between(episodes, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2)

    if epsilon_episode_marker > 0:
        vline = plt.axvline(x=epsilon_episode_marker, color='grey', linestyle=':', linewidth=2, label=f'Epsilon Min at Ep ~{epsilon_episode_marker}')
        plot_handles.append(vline)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('Episode')
    plt.grid(True, linestyle='--')
    plt.tight_layout()

    if save_filename is not None:
        save_path = os.path.join(base_drive_path, save_filename)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    plt.show()

# --- Plot 1: Mean Moving Average Scores ---
create_comparison_plot(
    metric_key='mean_moving_average_scores',
    std_metric_key='std_moving_average_scores',
    title='Comparison of Mean Moving Average Rewards (over last 100 episodes)',
    ylabel='Mean Moving Average Reward',
    epsilon_episode_marker=epsilon_min_episode,
    save_filename='comparison_mean_moving_average_rewards.png'
)

# --- Plot 2: Mean Cumulative Rewards ---
create_comparison_plot(
    metric_key='mean_cumulative_rewards',
    std_metric_key='std_cumulative_rewards',
    title='Comparison of Mean Cumulative Rewards',
    ylabel='Mean Cumulative Reward',
    epsilon_episode_marker=epsilon_min_episode,
    save_filename='comparison_mean_cumulative_rewards.png'
)