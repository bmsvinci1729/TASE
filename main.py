# main.py

import os
import time
from tase.perception.pose_estimator import get_pose_angles_from_image
from tase.simulation.humanoid_env import HumanoidWalkEnv

def main():
    """
    Main script to run the full pipeline:
    1. Process an image to get initial pose angles.
    2. Initialize the simulation environment with that pose.
    3. Run the simulation with random actions.
    """
    # --- Configuration ---
    MODEL_ASSET_PATH = os.path.join('assets', 'pose_landmarker_heavy.task')
    
    # its the model asset path, it is the path to the model asset file, it is a file that contains the model of the pose estimator
    TEST_IMAGE_PATH = os.path.join('images', '7.jpg')
    URDF_PATH = os.path.join('assets', 'humanoid.urdf')

    # --- Module 1: Perception ---
    print("[LOG] --- Running Perception Module ---")
    if not os.path.exists(TEST_IMAGE_PATH):
        print(f"[ERROR] Test image not found at {TEST_IMAGE_PATH}")
        return
        
    initial_angles = get_pose_angles_from_image(TEST_IMAGE_PATH, MODEL_ASSET_PATH)
    if not initial_angles:
        print("[ERROR] Could not get initial angles from the image. Exiting.")
        return
    print("[LOG] Successfully extracted initial angles from image.\n")

    # --- Module 2: Simulation ---
    print("[LOG] --- Initializing Simulation Module ---")
    env = HumanoidWalkEnv(urdf_path=URDF_PATH, render_mode='human')
    
    # Reset the environment with the pose from the image
    obs, info = env.reset(options={'initial_pose_angles': initial_angles})
    
    print("\n[LOG] --- Starting Interactive Simulation Loop ---")
    print("[LOG] Humanoid will take random actions until it falls.")
    
    start = time.time()
    while time.time() - start < 15:  # Run for 30 seconds or until termination
        # Get a random action from the environment's action space
        random_action = env.action_space.sample()
        
        # # Step the environment
        obs, reward, terminated, truncated, info = env.step(random_action)
        
        # Slow down the simulation for visualization
        time.sleep(1./240.)

    # Clean up
    env.close()
    print("[LOG] Simulation finished.")

if __name__ == '__main__':
    main()