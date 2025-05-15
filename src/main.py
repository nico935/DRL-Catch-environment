from environment import CatchEnv
from agent import DQNAgent, DDQNAgent
import numpy as np
import torch
import matplotlib.pyplot as plt
from network import NeuralNetwork, SmallerNeuralNetwork, DuelingNeuralNetwork # <--- Import both network classes
import math
import os 
import json 
import argparse
import time
NETWORK_CLASSES = {
    "NeuralNetwork": NeuralNetwork,
    "SmallerNeuralNetwork": SmallerNeuralNetwork,
    "DuelingNeuralNetwork": DuelingNeuralNetwork
}


# #setup the agent type
# DDQN=False
# DQN=True

# #setup the architecture type
# DuelingDQN=True
# USE_SMALL_NN=False


# if DDQN:
#     AGENT_TYPE = "DDQN"
#     if USE_SMALL_NN:
#         network_class = SmallerNeuralNetwork
#         print(f"Using network: {network_class} for {AGENT_TYPE}")
#     else:
#         network_class = NeuralNetwork
#         print(f"Using network: {network_class} for {AGENT_TYPE}")


# elif DQN:
#     AGENT_TYPE = "DQN"
#     if USE_SMALL_NN:
#         network_class = SmallerNeuralNetwork
#         print(f"Using network: {network_class} for {AGENT_TYPE}")
#     elif DuelingDQN:
#         network_class = DuelingNeuralNetwork
#         print(f"Using network: {network_class} for {AGENT_TYPE}")
#     else:
#         network_class = NeuralNetwork
#         print(f"Using network: {network_class} for {AGENT_TYPE}")


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config



def run_environment(seed_value, config,agent_class, network_class):  
    import time
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")


    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    #for reproducibility, slightly slows down training
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env = CatchEnv()
    state_dimensions = env.observation_space.shape
    n_actions = env.action_space.n

    agent_params = config["common_agent_params"].copy()
    if config["agent_type"] == "DDQN" and "ddqn_specific_params" in config:
        agent_params.update(config["ddqn_specific_params"])

    agent = agent_class(
        state_dimensions=state_dimensions,
        n_actions=n_actions,
        network_class=network_class,
        **agent_params 
    )

    print(f"--- Running Training for Seed: {seed_value} ---")
    print(f"Agent type: {config['agent_type']}")
    print(f"Network type: {config['network_architecture']}")
    print(f"Episodes: {config['n_episodes']}")
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
            print(f"episode {ep} | score: {score:.2f} | mov avg reward: {run_avg_score:.2f} | avg reward {avg_score:.2f} epsilon: {agent.epsilon:.2f} | steps: {agent.mem_counter}")

    EXPERIMENT_RESULTS_DIR = f"/content/drive/MyDrive/Courses Groning/Deep Reinforcement Learning/Assignent 1/{config['experiment_name']}"
    result_data = {
        "seed": seed_value,
        "config": config,
        "raw_scores": current_run_score, # current_run_score collected in the loop
        "moving_average_scores": current_run_moving_average, # current_run_moving_avg collected
        "epsilon_history": eps_history 
    }

    filename = f"results_seed_{seed_value}_{config['agent_type']}_{config['network_architecture']}_{run_timestamp}.json"
    filepath = os.path.join(EXPERIMENT_RESULTS_DIR, filename)

    if not os.path.exists(EXPERIMENT_RESULTS_DIR):
        os.makedirs(EXPERIMENT_RESULTS_DIR)

    with open(filepath, 'w') as f:
        json.dump(result_data, f, indent=4) # Use indent for readability
    print(f"Results for seed {seed_value} saved to {filepath}")

    return result_data


if __name__ == "__main__":
   # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Run DRL experiments using a configuration file.")
    parser.add_argument("config_file", type=str, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # --- Load Configuration ---
    config = load_config(args.config_file)

    # --- Calculate timestep where epsilon is constant ---
    epsilon_start = config["common_agent_params"]["epsilon_start"]
    epsilon_decay = config["common_agent_params"]["epsilon_decay"]
    epsilon_min = config["common_agent_params"]["epsilon_min"]
    
    burn_in_period_steps = config["common_agent_params"]["burn_in_period"] 
    AVG_STEPS_PER_EPISODE = 11 # As per your observation



    numerator = math.log(epsilon_min / epsilon_start)
    denominator = math.log(epsilon_decay)
    num_decay = math.ceil(numerator / denominator)
    constant_epsilon_episode = math.ceil((num_decay+burn_in_period_steps)/ AVG_STEPS_PER_EPISODE)
    print(f"Calculated: number of episodes till min epsilon reached: {constant_epsilon_episode}") # Optional

    # --- Determine Agent and Network Class from Config ---
    if config["agent_type"] == "DQN":
        agent_class = DQNAgent
    elif config["agent_type"] == "DDQN":
        agent_class = DDQNAgent
    else:
        raise ValueError(f"Unknown agent_type: {config['agent_type']}")

    network_class = NETWORK_CLASSES.get(config["network_architecture"])
    if network_class is None:
        raise ValueError(f"Unknown network_architecture: {config['network_architecture']}")

    # print(f"Selected Agent: {agent_class.__name__}, Network: {network_class.__name__}") # Optional

    N_EPISODES = config['n_episodes']
    SEEDS = config['seeds']

    # --- Run the Environment for each seed ---
    experiment_start_time = time.time()
    all_experiment_runs_data = []  
    for seed in SEEDS:
        result_from_seed = run_environment(seed, config, agent_class, network_class)
        all_experiment_runs_data.append(result_from_seed)
    # print(f"Total run time: {time.time() - experiment_start_time:.2f}s") # Optional

    # --- Process and Plot Results ---
    # Assuming all_experiment_runs_data is not empty and data is uniform
    EXPERIMENT_RESULTS_DIR = f"/content/drive/MyDrive/Courses Groning/Deep Reinforcement Learning/Assignent 1/{config['experiment_name']}"
    if not os.path.exists(EXPERIMENT_RESULTS_DIR):
        os.makedirs(EXPERIMENT_RESULTS_DIR)

    all_moving_averages = [run_data['moving_average_scores'] for run_data in all_experiment_runs_data]
    all_raw_scores = [run_data['raw_scores'] for run_data in all_experiment_runs_data]
    
    mean_ma_scores = np.mean(all_moving_averages, axis=0)
    std_ma_scores = np.std(all_moving_averages, axis=0)
    episodes_axis = np.arange(1, N_EPISODES + 1) # Assumes all runs go for N_EPISODES

    all_cumulative_rewards_list = [np.cumsum(scores) for scores in all_raw_scores]
    # Simple approach: Assume all episodes run to N_EPISODES for cumulative reward mean
    # If not, this np.mean will error or behave unexpectedly if inner lists have different lengths.
    # For the "simpler code" request, we omit padding that was in previous versions.
    mean_cumulative_rewards = np.mean(all_cumulative_rewards_list, axis=0)
    std_cumulative_rewards = np.std(all_cumulative_rewards_list, axis=0)
 

    fig, ax1 = plt.subplots(figsize=(12, 7)) # Adjusted size slightly
    experiment_timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Plot 1: Moving Average Reward
    color_ma = 'tab:blue'
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward Values')
    line1, = ax1.plot(episodes_axis, mean_ma_scores, color=color_ma, linestyle='-', label='Mean MA Reward')
    ax1.fill_between(episodes_axis, mean_ma_scores - std_ma_scores, mean_ma_scores + std_ma_scores, alpha=0.2, color=color_ma)
    ax1.tick_params(axis='y')

    # Plot 2: Cumulative Reward
    color_cum = 'tab:red'
    # Plotting against the same episodes_axis. Assumes mean_cumulative_rewards also has N_EPISODES length.
    line2, = ax1.plot(episodes_axis, mean_cumulative_rewards, color=color_cum, linestyle='--', label='Mean Cumulative Reward')
    ax1.fill_between(episodes_axis, 
                     mean_cumulative_rewards - std_cumulative_rewards, 
                     mean_cumulative_rewards + std_cumulative_rewards, 
                     alpha=0.2, color=color_cum)

    # Plot Vertical Line for Epsilon Convergence
    line3 = None
    if constant_epsilon_episode > 0:
        line3 = ax1.axvline(constant_epsilon_episode, color='tab:green', linestyle=':', linewidth=2, label=f'Epsilon Min at Ep ~{constant_epsilon_episode}')

    plt.title(f'{config["agent_type"]} ({config["network_architecture"]}) - Avg over {len(SEEDS)} seeds', pad=15)

    lines_for_legend = [line1, line2]
    if line3: 
        lines_for_legend.append(line3)
    
    labels_for_legend = [l.get_label() for l in lines_for_legend]
    ax1.legend(lines_for_legend, labels_for_legend, loc='best') # Simpler legend location
    
    ax1.grid(True, linestyle='--')
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect for title

    plot_filename = f"plot_{config['agent_type']}_{config['network_architecture']}_{experiment_timestamp}.png"
    saved_plot_path = os.path.join(EXPERIMENT_RESULTS_DIR, plot_filename)
    plt.savefig(saved_plot_path, bbox_inches='tight')
    print(f"Plot saved to: {saved_plot_path}")
    plt.show()
    

    