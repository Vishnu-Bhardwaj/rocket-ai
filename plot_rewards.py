# plot_rewards.py — Plot your AI's learning curve

import numpy as np
import matplotlib.pyplot as plt
import os

LOG_PATH = "./logs/evaluations.npz"

# Check the file exists
if not os.path.exists(LOG_PATH):
    print("❌ Could not find evaluations.npz")
    print("Make sure you trained using train.py with EvalCallback.")
else:
    # Load the evaluation data saved automatically during training
    data = np.load(LOG_PATH)

    timesteps = data["timesteps"]        # X axis — training steps
    results   = data["results"]          # Shape: (n_evals, n_eval_episodes)

    mean_rewards = results.mean(axis=1)  # Average reward at each checkpoint
    std_rewards  = results.std(axis=1)   # How consistent it was

    # ── Plot ──
    plt.figure(figsize=(12, 6))

    # Main reward line
    plt.plot(timesteps, mean_rewards,
             color="royalblue", linewidth=2.5, label="Mean Reward")

    # Shaded band showing ± standard deviation
    plt.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.25, color="royalblue", label="± Std Dev"
    )

    # "Solved" line at 200
    plt.axhline(y=200, color="green", linestyle="--",
                linewidth=2, label="Solved Threshold (200)")

    # "Random agent" baseline
    plt.axhline(y=-134, color="red", linestyle=":",
                linewidth=1.5, label="Random Agent Baseline (-134)")

    # Labels and styling
    plt.xlabel("Training Timesteps", fontsize=13)
    plt.ylabel("Mean Episode Reward", fontsize=13)
    plt.title("🚀 PPO Training Progress — Lunar Lander", fontsize=15)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save as image
    plt.savefig("training_curve.png", dpi=150)
    print("✅ Graph saved as training_curve.png")

    # Show the graph window
    plt.show()

    # Print a quick summary
    print(f"\n📊 Training Summary:")
    print(f"   Start reward : {mean_rewards[0]:.2f}")
    print(f"   End reward   : {mean_rewards[-1]:.2f}")
    print(f"   Peak reward  : {mean_rewards.max():.2f} "
          f"(at step {timesteps[mean_rewards.argmax()]:,})")