# explore.py — Understand the Lunar Lander environment

import gymnasium as gym
import numpy as np

# Create the Lunar Lander environment
# render_mode="human" opens a visual window
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to start a new episode
# 'state' is the initial observation (8 numbers describing the lander)
state, info = env.reset()

print("=== ENVIRONMENT INFO ===")
print(f"State space shape: {env.observation_space.shape}")
# The state has 8 values:
# [x_pos, y_pos, x_vel, y_vel, angle, angular_vel, left_leg_contact, right_leg_contact]

print(f"Number of possible actions: {env.action_space.n}")
# 4 actions: 0=nothing, 1=fire left, 2=fire main, 3=fire right

print(f"\nInitial state: {np.round(state, 3)}")
print("\n--- Running one random episode ---")

total_reward = 0
steps = 0

for _ in range(500):  # Run for up to 500 steps
    # Take a RANDOM action (the agent has no training yet — it's just flailing!)
    action = env.action_space.sample()
    
    # Apply the action to the environment
    # Returns: next_state, reward, terminated (crashed/landed), truncated (time limit), info
    next_state, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward
    steps += 1
    
    # Check if the episode ended
    if terminated or truncated:
        print(f"Episode ended after {steps} steps")
        print(f"Total reward: {total_reward:.2f}")
        break

env.close()  # Close the rendering window