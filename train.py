# train.py — Train a PPO agent to land the Lunar Lander

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os

# ──────────────────────────────────────────
# CONFIGURATION — Tweak these values
# ──────────────────────────────────────────
TOTAL_TIMESTEPS = 500_000     # How long to train (500k is a good start)
MODEL_SAVE_PATH = "lunar_lander_ppo"  # Where to save the model
LOG_DIR = "./logs/"            # Where to save training logs
N_ENVS = 4                     # Run 4 environments in parallel (faster training)
EVAL_FREQ = 10_000             # Evaluate every 10,000 steps
N_EVAL_EPISODES = 20           # Use 20 episodes for evaluation

# ──────────────────────────────────────────
# STEP 1: Create training environments
# ──────────────────────────────────────────
# make_vec_env creates multiple parallel environments for faster data collection
# No render_mode here — rendering slows training down
train_env = make_vec_env("LunarLander-v3", n_envs=N_ENVS)

# Create a separate environment for evaluation (just 1, no parallelism needed)
eval_env = gym.make("LunarLander-v3")

# ──────────────────────────────────────────
# STEP 2: Create the PPO model
# ──────────────────────────────────────────
model = PPO(
    policy="MlpPolicy",     # MlpPolicy = a standard neural network (Multi-Layer Perceptron)
    env=train_env,
    n_steps=1024,           # Collect 1024 steps per environment before updating
    batch_size=64,          # Process 64 transitions at a time during updates
    n_epochs=4,             # Number of passes over the collected data
    gamma=0.999,            # Discount factor: how much future rewards matter (0-1)
                            # 0.999 means the agent cares a lot about the future
    gae_lambda=0.98,        # Controls the bias-variance tradeoff in advantage estimation
    ent_coef=0.01,          # Entropy bonus: encourages exploration (trying new things)
    verbose=1,              # Print training progress
    tensorboard_log=LOG_DIR # Save logs for TensorBoard visualization
)

print(f"Model policy network:\n{model.policy}")
print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps across {N_ENVS} parallel environments...")
print("This may take 10-30 minutes on a standard laptop.\n")

# ──────────────────────────────────────────
# STEP 3: Set up callbacks (automated checkpoints and evaluation)
# ──────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)

# EvalCallback: automatically evaluates the agent every EVAL_FREQ steps
# and saves the best model found so far
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path=LOG_DIR + "best_model/",
    log_path=LOG_DIR,
    eval_freq=EVAL_FREQ,
    n_eval_episodes=N_EVAL_EPISODES,
    deterministic=True,   # Use greedy policy during evaluation (no random exploration)
    render=False
)

# CheckpointCallback: saves the model at regular intervals as a backup
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=LOG_DIR + "checkpoints/",
    name_prefix="lunar_ppo"
)

# ──────────────────────────────────────────
# STEP 4: Train!
# ──────────────────────────────────────────
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[eval_callback, checkpoint_callback],
    progress_bar=True  # Shows a nice progress bar (requires tqdm: pip install tqdm)
)

# ──────────────────────────────────────────
# STEP 5: Save the final model
# ──────────────────────────────────────────
model.save(MODEL_SAVE_PATH)
print(f"\n✅ Training complete! Model saved to '{MODEL_SAVE_PATH}.zip'")
print(f"✅ Best model saved to '{LOG_DIR}best_model/'")

train_env.close()
eval_env.close()