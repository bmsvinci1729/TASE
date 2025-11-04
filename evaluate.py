import torch
import numpy as np
import pickle
import random
import os
import time

# Import our environment
from tase.simulation.humanoid_env import HumanoidWalkEnv

# --- IMPORTANT ---
# We only need the network architecture (QNetwork)
# and the action-selection function (select_action)
# from your q_network.py file.[1]
from tase.rl.q_network import QNetwork, select_action, INPUT_DIM

# --- Configuration ---
URDF_PATH = os.path.join('assets', 'humanoid.urdf')
POSE_LIBRARY_PATH = 'pose_library.pkl'
MODEL_SAVE_PATH = 'dqn_humanoid.pth' # The file you just created
NUM_TEST_EPISODES = 10  # How many different poses to test
MAX_T = 1000             # Max steps per episode

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def evaluate():
    """
    Loads a trained DQN model and runs it in the visual
    environment to evaluate its performance.
    """
    
    # --- 1. Load the Trained Model ---
    print(f"Loading trained model from {MODEL_SAVE_PATH}...")
    
    # We use the same QNetwork architecture [1]
    model = QNetwork().to(device)
    
    # Load the saved weights from your.pth file
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    # Set the model to evaluation mode (this is important!)
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Load the Pose Library ---
    print(f"Loading pose library from {POSE_LIBRARY_PATH}...")
    try:
        with open(POSE_LIBRARY_PATH, 'rb') as f:
            pose_library = pickle.load(f)
    except FileNotFoundError:
        print(f"*** ERROR: Pose library '{POSE_LIBRARY_PATH}' not found.")
        print("Please run 'build_pose_library.py' first.")
        return
    print(f"Loaded {len(pose_library)} poses.")

    # --- 3. Initialize the *VISUAL* Environment ---
    # We use 'human' mode now so we can see the robot [1]
    print("Initializing visual environment...")
    env = HumanoidWalkEnv(urdf_path=URDF_PATH, render_mode='human')
    
    # --- 4. Run Evaluation Loop ---
    for i_episode in range(1, NUM_TEST_EPISODES + 1):
        
        # Select a random starting pose
        initial_pose = random.choice(pose_library) [2, 3]
        print(f"\n--- Episode {i_episode}/{NUM_TEST_EPISODES} ---")
        
        # Reset the environment with that pose
        state, _ = env.reset(options={'initial_pose_angles': initial_pose}) [1, 1]
        
        episode_reward = 0
        
        for t in range(MAX_T):
            # 1. Convert state to a tensor
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            
            # 2. Get Q-values from the loaded model
            with torch.no_grad():
                q_values = model(state_tensor)
            
            # 3. Select the BEST action (no epsilon)
            # We use the select_action function from q_network.py [1]
            action_torques = select_action(q_values)
            
            # 4. Step the environment
            next_state, reward, terminated, truncated, _ = env.step(action_torques)
            
            state = next_state
            episode_reward += reward
            
            # We MUST sleep here to make the rendering visible [1]
            time.sleep(1./240.)
            
            if terminated or truncated:
                print(f"Episode finished after {t+1} steps. Reward: {episode_reward:.2f}")
                break

    # Clean up
    env.close()
    print("\n--- Evaluation Complete ---")


if __name__ == '__main__':
    evaluate()