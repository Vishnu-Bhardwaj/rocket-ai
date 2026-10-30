# compare_algorithms.py — PPO vs A2C side by side

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
import os

# ── Config ──
TOTAL_TIMESTEPS = 300_000   # 300k each (faster than 500k, good enough to compare)
EVAL_FREQ      = 10_000
N_EVAL_EPS     = 20
N_ENVS         = 4

algorithms = {
    "PPO": PPO,
    "A2C": A2C,
}

all_results = {}   # Store results for plotting

# ── Train each algorithm one by one ──
for name, AlgoClass in algorithms.items():
    print(f"\n{'='*45}")
    print(f"  Training {name} for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"{'='*45}\n")

    log_dir = f"./logs_compare/{name}/"
    os.makedirs(log_dir, exist_ok=True)

    # Create environments
    train_env = make_vec_env("LunarLander-v3", n_envs=N_ENVS)
    eval_env  = gym.make("LunarLander-v3")

    # Create model
    model = AlgoClass(
        "MlpPolicy",
        train_env,
        verbose=0,               # Quiet mode — only eval results shown
        tensorboard_log=log_dir,
        gamma=0.999,
    )

    # EvalCallback saves scores at each checkpoint
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir + "best_model/",
        log_path=log_dir,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPS,
        deterministic=True,
        render=False,
        verbose=1
    )

    # Train!
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

    # Save the final model
    model.save(f"./logs_compare/{name}/final_model")

    # Load evaluation results
    eval_data = np.load(log_dir + "evaluations.npz")
    all_results[name] = {
        "timesteps"    : eval_data["timesteps"],
        "mean_rewards" : eval_data["results"].mean(axis=1),
        "std_rewards"  : eval_data["results"].std(axis=1),
        "final_mean"   : eval_data["results"].mean(axis=1)[-1],
        "peak_mean"    : eval_data["results"].mean(axis=1).max(),
    }

    print(f"\n✅ {name} done! Final reward: {all_results[name]['final_mean']:.2f}")

    train_env.close()
    eval_env.close()

# ── Print Summary Table ──
print(f"\n{'='*45}")
print(f"  FINAL COMPARISON")
print(f"{'='*45}")
print(f"{'Algorithm':<10} {'Final Reward':>14} {'Peak Reward':>12}")
print(f"{'-'*40}")
for name, res in all_results.items():
    print(f"{name:<10} {res['final_mean']:>14.2f} {res['peak_mean']:>12.2f}")
print(f"\n✅ Winner: {max(all_results, key=lambda n: all_results[n]['final_mean'])}")

# ── Plot both curves on one graph ──
colors = {"PPO": "royalblue", "A2C": "tomato"}

plt.figure(figsize=(13, 6))

for name, res in all_results.items():
    plt.plot(
        res["timesteps"], res["mean_rewards"],
        label=f"{name} (peak: {res['peak_mean']:.0f})",
        color=colors[name], linewidth=2.5
    )
    plt.fill_between(
        res["timesteps"],
        res["mean_rewards"] - res["std_rewards"],
        res["mean_rewards"] + res["std_rewards"],
        alpha=0.2, color=colors[name]
    )

# Reference lines
plt.axhline(y=200,  color="green", linestyle="--", linewidth=2,   label="Solved (200)")
plt.axhline(y=-134, color="gray",  linestyle=":",  linewidth=1.5, label="Random baseline (-134)")

plt.xlabel("Training Timesteps", fontsize=13)
plt.ylabel("Mean Episode Reward", fontsize=13)
plt.title("🚀 PPO vs A2C — Lunar Lander Training Comparison", fontsize=15)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("comparison_curve.png", dpi=150)
print("\n📊 Graph saved as comparison_curve.png")
plt.show()