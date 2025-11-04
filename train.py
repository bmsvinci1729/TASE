import torch
import numpy as np
import pickle
import random
from collections import deque
import os

# Import our custom classes
from tase.simulation.humanoid_env import HumanoidWalkEnv
from tase.rl.dqn_agent import DQNAgent

# --- Configuration ---
URDF_PATH = os.path.join('assets', 'humanoid.urdf')
POSE_LIBRARY_PATH = 'pose_library.pkl'
MODEL_SAVE_PATH = 'dqn_humanoid.pth'
SEED = 0

# --- Training Hyperparameters ---
NUM_EPISODES = 1000       # Total episodes to run
MAX_T = 1000              # Max steps per episode
EPS_START = 1.0           # Starting epsilon for exploration
EPS_END = 0.01            # Minimum epsilon
EPS_DECAY = 0.995         # Multiplicative decay (per episode)

def train():
    """Main training loop for the TASE project."""
    
    # 1. Load Pose Library (Final Module - Task 1) 
    print(f"Loading pose library from {POSE_LIBRARY_PATH}...")
    try:
        with open(POSE_LIBRARY_PATH, 'rb') as f:
            pose_library = pickle.load(f)
        if not pose_library:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"*** ERROR: Pose library '{POSE_LIBRARY_PATH}' not found or empty.")
        print("Please run 'build_pose_library.py' first.")
        return
    print(f"Loaded {len(pose_library)} poses.")

    # 2. Initialize Environment 
    # Note: render_mode='direct' for fast, non-visual training.
    env = HumanoidWalkEnv(urdf_path=URDF_PATH, render_mode='direct')
    env.reset(seed=SEED)
    
    # 3. Initialize Agent 
    agent = DQNAgent(seed=SEED)
    print(f"DQN Agent initialized. Using device: {agent.policy_net.fc1.weight.device}")

    # 4. Initialize Training Variables
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    epsilon = EPS_START                # initialize epsilon

    print("--- Starting Training ---")
    
    # 5. Main Training Loop
    for i_episode in range(1, NUM_EPISODES + 1):
        
        # --- Pose Randomization (Final Module - Task 2) ---
        # Select a random initial pose from the library 
        initial_pose = random.choice(pose_library)
        
        # Reset the environment with that pose
        state, _ = env.reset(options={'initial_pose_angles': initial_pose})
        
        episode_reward = 0
        for t in range(MAX_T):
            # 1. Agent selects an action (epsilon-greedy)
            action_indices, action_torques = agent.select_action_epsilon_greedy(state, epsilon)
            
            # 2. Environment steps
            next_state, reward, terminated, truncated, _ = env.step(action_torques)
            
            # 3. Agent stores experience and learns
            #    (The learn() method is called inside agent.step()) 
            agent.step(state, action_indices, reward, next_state, terminated)
            
            state = next_state
            episode_reward += reward
            if terminated or truncated:
                break
        
        # --- End of Episode ---
        scores_window.append(episode_reward)
        scores.append(episode_reward)
        
        # Decay epsilon
        epsilon = max(EPS_END, EPS_DECAY * epsilon)
        
        # Print progress
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {epsilon:.3f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {epsilon:.3f}')
            # Save model checkpoint (Expected Output 1) 
            torch.save(agent.policy_net.state_dict(), MODEL_SAVE_PATH)
            print(f"  [LOG] Model checkpoint saved to {MODEL_SAVE_PATH}")

    # --- End of training ---
    env.close()
    print("\n--- Training Complete ---")
    torch.save(agent.policy_net.state_dict(), MODEL_SAVE_PATH)
    print(f"Final model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()