# watch.py — Render the trained agent landing the spacecraft

import gymnasium as gym
from stable_baselines3 import PPO
import time

# ──────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────
MODEL_PATH = "logs/best_model/best_model"  # Path to the trained model
N_EPISODES = 10                             # Number of landings to watch

# Create the environment with visual rendering
env = gym.make("LunarLander-v3", render_mode="human")

# Load the trained model
# Note: We pass the environment to validate compatibility
model = PPO.load(MODEL_PATH, env=env)

print(f"Watching {N_EPISODES} landing attempts...\n")
print("(Close the window or press Ctrl+C to stop)\n")

episode_rewards = []

for episode in range(1, N_EPISODES + 1):
    # Reset: start a new episode
    state, info = env.reset()
    total_reward = 0
    steps = 0
    done = False
    
    while not done:
        # The agent decides what action to take based on the current state
        # deterministic=True means always pick the best action (no random exploration)
        action, _ = model.predict(state, deterministic=True)
        
        # Apply the action
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
        
        # Small pause for smooth rendering (optional)
        time.sleep(0.01)
    
    episode_rewards.append(total_reward)
    result = "✅ LANDED" if total_reward > 200 else ("💥 CRASHED" if total_reward < 0 else "⚠️ OK")
    print(f"Episode {episode:2d}: {result} | Reward: {total_reward:7.2f} | Steps: {steps}")

env.close()

# Summary
avg = sum(episode_rewards) / len(episode_rewards)
print(f"\n📊 Summary over {N_EPISODES} episodes:")
print(f"   Average Reward: {avg:.2f}")
print(f"   Best:  {max(episode_rewards):.2f}")
print(f"   Worst: {min(episode_rewards):.2f}")