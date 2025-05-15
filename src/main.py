from environment import CatchEnv
from agent import DQNAgent, DDQNAgent
import numpy as np
import torch
import matplotlib.pyplot as plt
import os 
import json 


N_EPISODES = 800
DDQN=False
DQN=True
#SEEDS= [42, 123, 789, 101, 555]
SEEDS= [42, 42]
all_runs_scores = []
all_runs_moving_averages = []
if DDQN:
    AGENT_TYPE = "DDQN"
    RESULTS_DIR = "catch_ddqn_results"
elif DQN:
    AGENT_TYPE = "DQN"
    RESULTS_DIR = "catch_dqn_results"

DRIVE_RESULTS_PATH = f"/content/drive/MyDrive/Courses Groning/Deep Reinforcement Learning/Assignent 1/{AGENT_TYPE}_results"

RESULTS_DIR = DRIVE_RESULTS_PATH

if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)
    print(f"Created results directory in Google Drive: {RESULTS_DIR}")

def run_environment(seed_value):  
    import time
    timestamp = time.strftime("%Y%m%d-%M%S")

    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    env = CatchEnv()
    state_dimensions = env.observation_space.shape
    n_actions = env.action_space.n
    if DQN==True:
        agent = DQNAgent(
            memory_size=10000,     
            state_dimensions=state_dimensions,
            n_actions=n_actions,
            learning_rate=0.00025,  
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_min=0.0001,
            epsilon_decay=0.999901,   
            batch_size=32,
            target_update_frequency=2000 
        )
    if DDQN==True:
        agent = DDQNAgent(
            memory_size=10000,      
            state_dimensions=state_dimensions,
            n_actions=n_actions,
            learning_rate=0.00025,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_min=0.0001,
            epsilon_decay=0.999901,  
            t_weight_start=0.005,
            t_weight_min=0.00001,
            t_weight_decay=0.999,
            target_update_frequency=2000,
            soft_update=False,
        )
    print(f"Agent type: {AGENT_TYPE}")
    print(f"Episodes: {N_EPISODES}")
    print(f"learning rate: {agent.lr}")
    print(f"Using deviiice: {agent.device}")
    print(f"State dimensionnns: {state_dimensions}")
    print(f"Number of actions: {n_actions}")
    print(f"--- Running Training for Seed: {seed_value} ---")



    eps_history = []
    current_run_score=[]
    current_run_moving_average=[]
    
    for ep in range(N_EPISODES):
        observation, info = env.reset(seed=seed_value+ep)
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
    result_data = {
        "seed": seed_value,
        "raw_scores": current_run_score, # current_run_score collected in the loop
        "moving_average_scores": current_run_moving_average # current_run_moving_avg collected
    }
    # Save data for this seed
    filepath = os.path.join(RESULTS_DIR, f"results_seed_{seed_value}_{timestamp}.json")
    with open(filepath, 'w') as f:
        json.dump(result_data, f)
    print(f"Results for seed {seed_value} saved to {filepath}")
    return current_run_score, current_run_moving_average

if __name__ == "__main__":
    import time
    timestamp = time.strftime("%M%S")
    for seed in SEEDS:
        raw_scores, moving_avg_scores = run_environment(seed)
        all_runs_scores.append(raw_scores)
        all_runs_moving_averages.append(moving_avg_scores)

    # --- Process and Plot Results ---
    mean_ma_scores = np.mean(all_runs_moving_averages, axis=0)
    std_ma_scores = np.std(all_runs_moving_averages, axis=0)
   
    episodes_axis = np.arange(1, N_EPISODES + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_axis, mean_ma_scores, label='Mean Moving Average Reward (across seeds)')

    # shaded region for +/- 1 sd
    plt.fill_between(episodes_axis,
                     mean_ma_scores - std_ma_scores,
                     mean_ma_scores + std_ma_scores,
                     alpha=0.2, label='Mean +/- 1 Std Dev')

    # shaded region for +/- 2 sd
    # plt.fill_between(episodes_axis,
    #                  mean_ma_scores - 2 * std_ma_scores,
    #                  mean_ma_scores + 2 * std_ma_scores,
    #                  alpha=0.1, label='Mean +/- 2 Std Dev')

    plt.title(f'{AGENT_TYPE} Perforcmance on Catch Game (Average over {len(SEEDS)} seeds)')
    plt.xlabel('Episode')
    plt.ylabel('100-Episode Moving Average Reward')
    plt.legend()
    plt.grid(True)
    plot_filename = f"{AGENT_TYPE}_{timestamp}_plot.png"
    saved_plot_path = os.path.join(RESULTS_DIR, plot_filename)
    if os.path.exists(saved_plot_path):
        display(Image(filename=saved_plot_path))
    else:
        print(f"Plot image not found at: {saved_plot_path}")


