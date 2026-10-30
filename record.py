# record_video.py — Record your AI landing as an MP4 video

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
import os

MODEL_PATH = "logs/best_model/best_model"
VIDEO_FOLDER = "./videos/"
N_EPISODES = 5  # Record 5 landings

# Create the videos folder
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# render_mode MUST be "rgb_array" for recording (not "human")
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Wrap with RecordVideo — records every episode automatically
env = RecordVideo(
    env,
    video_folder=VIDEO_FOLDER,
    episode_trigger=lambda ep: True,  # Record ALL episodes
    name_prefix="lunar-lander"        # Video file name prefix
)

# Load your trained model
model = PPO.load(MODEL_PATH, env=env)

print(f"🎬 Recording {N_EPISODES} landing attempts...")
print(f"Videos will be saved to: {VIDEO_FOLDER}\n")

for episode in range(N_EPISODES):
    state, _ = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    result = "✅ LANDED" if total_reward >= 200 else "💥 CRASHED"
    print(f"Episode {episode + 1}: {result} | Score: {total_reward:.2f}")

env.close()
print(f"\n✅ Done! Open the 'videos' folder in your project to find the MP4 files.")