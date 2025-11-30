"""Run a trained PPO agent; optionally initialize humanoid pose from an image.

Usage: prompts for an image path, attempts pose extraction and state initialization,
then runs the agent in a human-rendered environment.
"""

import gymnasium as gym
import numpy as np
import torch
import glfw
import mujoco

from agent.rl_agent import PPOAgent
from pose_estimation import load_image, preprocess_image, PoseExtractor, body25_to_humanoid_pose


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("Humanoid-v5", render_mode="human")

    # Optionally initialize environment state from an input image
    print("Input file name for initial pose (or press Enter to skip): ")
    image_path = input().strip()
    out_path = "out.jpg"
    try:
        if image_path:
            image = load_image(image_path)
            preprocessed_image = preprocess_image(image)
            pose_extractor = PoseExtractor()
            body25 = pose_extractor.extract_keypoints(preprocessed_image)
            if len(body25) > 0:
                pose = body25_to_humanoid_pose(body25)
                pose_extractor.draw_skeleton(image, body25, out_path)
                obs, _ = env.reset()
                qpos = env.unwrapped.data.qpos.copy()
                qvel = env.unwrapped.data.qvel.copy()
                qpos[18:21] = pose[11:14]
                qpos[21:24] = pose[14:17]
                env.unwrapped.set_state(qpos, qvel)
                obs = env.unwrapped._get_obs()
            else:
                print("No pose detected; using default reset state.")
                obs, _ = env.reset()
        else:
            obs, _ = env.reset()
    except FileNotFoundError:
        print(f"Image not found at '{image_path}'. Using default reset state.")
        obs, _ = env.reset()

    obs_dim = env.observation_space.shape
    action_dim = env.action_space.shape

    agent = PPOAgent(obs_dim[0], action_dim[0]).to(device)
    agent.load_state_dict(torch.load("model.pt", map_location=torch.device('cpu')))
    agent.eval()

    # Configure viewer camera
    env.render()
    viewer = env.unwrapped.mujoco_renderer.viewer
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = 0
    glfw.maximize_window(viewer.window)

    # Lookup body IDs once
    torso_id = env.unwrapped.model.body("torso").id

    # Constant external force to apply to torso (B, x-axis)
    constant_force = np.array([6, 0.0, 0.0, 0.0, 0.0, 0.0])

    done = False
    while not done:
        env.render()
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.tensor(np.array([obs], dtype=np.float32), device=device))
        action_np = action.squeeze(0).cpu().numpy()
        obs, _, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

    # Example loop: step while applying external torso force
    done = False
    while not done:
        env.render()
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.tensor(np.array([obs], dtype=np.float32), device=device))
        env.unwrapped.data.xfrc_applied[torso_id] = constant_force
        obs, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated

    env.close()


if __name__ == "__main__":
    main()
