import argparse

import cv2
import gymnasium as gym
import numpy as np
import torch


def parse_args_ppo() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True if torch.cuda.is_available() else False, action="store_true",
                        help="Use CUDA")
    parser.add_argument("--env", default="Humanoid-v5", help="Environment to use")
    parser.add_argument("--n-envs", type=int, default=32, help="Number of environments")
    parser.add_argument("--n-epochs", type=int, default=1000, help="Number of epochs to run")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps per epoch per environment")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--train-iters", type=int, default=10, help="Number of training iterations")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Lambda for GAE")
    parser.add_argument("--clip-ratio", type=float, default=0.2, help="PPO clip ratio")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="Entropy coefficient")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--target-kl", type=float, default=0.01, help="Target KL divergence")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--reward-scale", type=float, default=0.01, help="Reward scaling")
    parser.add_argument("--render-epoch", type=int, default=1000, help="Render every n-th epoch")
    return parser.parse_args()


def log_video(env, agent, device, video_path, fps=30):
    """Record a single episode and save as mp4.

    Args:
      - env: single-env instance with `render()` and `step()`.
      - agent: model with `get_action_and_value()`.
      - device: torch device for inputs.
      - video_path: output filepath.
    """
    agent.eval()
    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        frames.append(env.render())
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(torch.tensor(np.array([obs], dtype=np.float32), device=device))
        obs, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated

    # write mp4 (OpenCV expects BGR)
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()


def make_env(env_id, reward_scaling=0.01, render=False, fps=30):
    """Create and return a gym environment wrapped to scale rewards.

    If render=True, the env is created with 'rgb_array' mode and its internal fps set.
    """
    if render:
        env = gym.make(env_id, render_mode='rgb_array')
        env.metadata['render_fps'] = fps
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    else:
        env = gym.make(env_id)
        env = gym.wrappers.TransformReward(env, lambda r: r * reward_scaling)
    return env
