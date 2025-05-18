from environment import CatchEnv
from agent import DQNAgent, DDQNAgent, DQVAgent
import numpy as np
import torch
import matplotlib.pyplot as plt
from network import QNetwork, SmallQNetwork, DuelingNeuralNetwork , VNetwork
import math
import os 
import json 
import argparse
import time



def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def run_environment(seed_value, config,agent_class,network_args):  
    # Set random seeds for reproducibility
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    #Make sure torch calculation are deterministic, slightly slows down training
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env = CatchEnv()
    state_dimensions = env.observation_space.shape
    n_actions = env.action_space.n

    agent_params = config.copy()

    agent = agent_class(
        state_dimensions=state_dimensions,
        n_actions=n_actions,
        **network_args,
        **agent_params
    )
   
    print(f"--- Running Training for Seed: {seed_value} ---")
    print(f"Agent type: {config['agent_type']}")
    print(f"Learning rate: {agent.lr}") 
    print(f"Using device: {agent.device}") 
    print(f"State dimensions: {state_dimensions}")
    print(f"Number of actions: {n_actions}")



    eps_history = []
    current_run_score=[]
    current_run_moving_average=[]

    N_EPISODES = config['n_episodes']

    for ep in range(N_EPISODES):
        if ep==0:
            observation, info = env.reset(seed=seed_value)
        else:
            observation, info = env.reset()

        terminated = False
        truncated = False 
        score = 0

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
            print(f"score: {score:.2f} | mov avg reward: {run_avg_score:.2f} | avg reward {avg_score:.2f} epsilon: {agent.epsilon:.2f} | steps: {agent.mem_counter}")

    result_data = {
        "seed": seed_value,
        "config": config,
        "raw_scores": current_run_score, # current_run_score collected in the loop
        "moving_average_scores": current_run_moving_average, # current_run_moving_avg collected
        "epsilon_history": eps_history 
    }

    return result_data


if __name__ == "__main__":
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run DRL experiments using a configuration file.")
    parser.add_argument("config_file", type=str, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # --- Load Configuration ---
    config = load_config(args.config_file)
    print(f"Configuration: {config}")

    N_EPISODES = config['n_episodes']
    SEEDS = config['seeds']
    # --- Define and create results directory ---
    EXPERIMENT_RESULTS_DIR = f"/content/drive/MyDrive/Courses Groning/Deep Reinforcement Learning/Assignent 1/{config['experiment_name']}"
    if not os.path.exists(EXPERIMENT_RESULTS_DIR):
        os.makedirs(EXPERIMENT_RESULTS_DIR)

    NETWORK_CLASSES = {
    "QNetwork": QNetwork,
    "SmallQNetwork": SmallQNetwork,
    "DuelingNeuralNetwork": DuelingNeuralNetwork,
    "ValueNetwork": VNetwork,  
    }

    # --- Build network_args and net_name ---

    # Extract network classes from config
    network_args = {
        key: NETWORK_CLASSES[config[key]]
        for key in config
        if key.endswith("_class")
    }
    # We remove the keys in config to eliminate overlap with network_args
    for key in network_args:
        config.pop(key, None)

    # Remove the "_class" suffix to assign a string name to network used
    net_name = "_".join([v.__name__ for v in network_args.values()])


    # --- Define the agent class and agent name ---

    AGENT_CLASSES = {
        "DQN": DQNAgent,
        "DDQN": DDQNAgent,
        "DQV": DQVAgent,
        "DQVMax": DQVMaxAgent,
    }

    agent_class = AGENT_CLASSES.get(config["agent_type"])
    if agent_class is None:
        raise ValueError(f"Unknown agent_type: {config['agent_type']}")

    print(f"Selected Agent: {agent_class.__name__}, Network: {net_name}")


    # --- Calculate timestep where epsilon is constant ---
    epsilon_start = config["epsilon_start"]
    epsilon_decay = config["epsilon_decay"]
    epsilon_min = config["epsilon_min"]
    
    burn_in_period_steps = config["burn_in_period"] 
    AVG_STEPS_PER_EPISODE = 11 # ball drops in 11 steps

    numerator = math.log(epsilon_min / epsilon_start)
    denominator = math.log(epsilon_decay)
    num_decay = math.ceil(numerator / denominator)
    constant_epsilon_episode = math.ceil((num_decay+burn_in_period_steps)/ AVG_STEPS_PER_EPISODE)
    print(f"Calculated: number of episodes till min epsilon reached: {constant_epsilon_episode}") 


    # --- Run the Environment for each seed ---
    start_time = time.time()
    all_experiment_runs_data = []  
    for seed in SEEDS:
        result_from_seed = run_environment(seed, config, agent_class,network_args)
        all_experiment_runs_data.append(result_from_seed)
    end_time = time.time()
    print(f"Total run time: {end_time- start_time:.2f}s")

    all_moving_averages = [run_data['moving_average_scores'] for run_data in all_experiment_runs_data]
    all_raw_scores = [run_data['raw_scores'] for run_data in all_experiment_runs_data]
    
    mean_ma_scores = np.mean(all_moving_averages, axis=0)
    std_ma_scores = np.std(all_moving_averages, axis=0)
    episodes_axis = np.arange(1, N_EPISODES + 1) 

    all_cumulative_rewards_list = [np.cumsum(scores) for scores in all_raw_scores]
    mean_cumulative_rewards = np.mean(all_cumulative_rewards_list, axis=0)
    std_cumulative_rewards = np.std(all_cumulative_rewards_list, axis=0)

    # --- Save Experiment Results ---
    experiment_results = {
    "config": config,
    "mean_moving_average_scores": mean_ma_scores.tolist(),
    "std_moving_average_scores": std_ma_scores.tolist(),
    "mean_cumulative_rewards": mean_cumulative_rewards.tolist(),
    "std_cumulative_rewards": std_cumulative_rewards.tolist(),
    "all_seeds": SEEDS
    }

    filename = f"results_{config['agent_type']}_{net_name}.json"
    filepath = os.path.join(EXPERIMENT_RESULTS_DIR, filename)

    with open(filepath, 'w') as f:
        json.dump(experiment_results, f, indent=4) 
    print(f"Results saved to {filepath}")


    # --- Create Two Plots ---
    fig, axs = plt.subplots(2, 1, figsize=(12, 14), sharex=True) # 2 rows, 1 column, share x-axis
    experiment_timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Plot 1: Moving Average Reward
    color_ma = 'tab:blue'
    axs[0].set_ylabel('Mean Average Reward (last min(t,100) episodes')
    axs[0].set_xlabel('Episode')
    axs[0].plot(episodes_axis, mean_ma_scores, color=color_ma, linestyle='-', label='Mean MA Reward')
    axs[0].fill_between(episodes_axis, mean_ma_scores - std_ma_scores, mean_ma_scores + std_ma_scores, alpha=0.2, color=color_ma)

    # Draw vertical line for constant epsilon
    line_eps_ma = None # Initialize for legend
    if constant_epsilon_episode > 0 and constant_epsilon_episode <= N_EPISODES : # Check if within plot range
        line_eps_ma = axs[0].axvline(constant_epsilon_episode, color='tab:green', linestyle=':', linewidth=2, label=f'Epsilon Min at Ep ~{constant_epsilon_episode}')
    
    # Set title
    axs[0].set_title(f'{config["agent_type"]} ({net_name}) - Avg over {len(SEEDS)} seeds')
    
    # Set legend
    legend_elements_ma = [axs[0].get_lines()[0]]
    if line_eps_ma: 
        legend_elements_ma.append(line_eps_ma)
    axs[0].legend(handles=legend_elements_ma, loc='best')
    axs[0].grid(True, linestyle='--')

    # Plot 2: Cumulative Reward
    color_cum = 'tab:red'
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Mean Cumulative Reward')
    axs[1].plot(episodes_axis, mean_cumulative_rewards, color=color_cum, linestyle='--', label='Mean Cumulative Reward')
    axs[1].fill_between(episodes_axis, 
                       mean_cumulative_rewards - std_cumulative_rewards, 
                       mean_cumulative_rewards + std_cumulative_rewards, 
                       alpha=0.2, color=color_cum)

    line_eps_cum = None 
    if constant_epsilon_episode > 0 and constant_epsilon_episode <= N_EPISODES: 
        line_eps_cum = axs[1].axvline(constant_epsilon_episode, color='tab:green', linestyle=':', linewidth=2, label=f'Epsilon Min at Ep ~{constant_epsilon_episode}')
    
    legend_elements_cum = [axs[1].get_lines()[0]] # 
    if line_eps_cum: 
        legend_elements_cum.append(line_eps_cum)
    axs[1].legend(handles=legend_elements_cum, loc='best')
    axs[1].grid(True, linestyle='--')
    
    fig.tight_layout(rect=[0, 0.03, 1, 0.96]) 

    plot_filename = f"plot_MA_Cum_{config['agent_type']}_{net_name}_{experiment_timestamp}.png"
    saved_plot_path = os.path.join(EXPERIMENT_RESULTS_DIR, plot_filename)
    plt.savefig(saved_plot_path, bbox_inches='tight')
    print(f"Plot saved to: {saved_plot_path}")
    plt.show()
    

    