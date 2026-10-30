# 🚀 AI Lands a Rocket — Reinforcement Learning with Lunar Lander

A beginner-friendly reinforcement learning project where an AI agent learns
to safely land a spacecraft using **PPO (Proximal Policy Optimization)**.

Built with Python, Gymnasium, and Stable-Baselines3.

---

## 📊 Results

| Agent | Mean Reward | Status |
|-------|------------|--------|
| Random (untrained) | -134 | 💥 Crashes every time |
| PPO (500k steps) | **204** | ✅ Solved! |
| A2C (300k steps) | 112 | ⚠️ Partially trained |

> The environment is considered **solved** at a mean reward of 200+.

---

## 🧠 What is this?

The agent controls a Lunar Lander spacecraft with 4 actions:
- Do nothing
- Fire left engine
- Fire main engine
- Fire right engine

It learns purely through **trial and error** — no hand-coded rules.

---

## 🛠️ Setup

### Requirements
- Python 3.10 or 3.11
- Windows / macOS / Linux

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/rocket-ai.git
cd rocket-ai

# Create virtual environment
python -m venv rocketenv
.\rocketenv\Scripts\Activate.ps1   # Windows
# source rocketenv/bin/activate    # macOS/Linux

# Install dependencies
pip install swig
pip install gymnasium[box2d] stable-baselines3[extra] pygame matplotlib moviepy
```

---

## 🚀 Usage

### 1. Explore the environment (random agent)
```bash
python explore.py
```

### 2. Train the PPO agent (~45 mins on CPU)
```bash
python train.py
```

### 3. Watch the trained AI land
```bash
python watch.py
```

### 4. Record a video
```bash
python record_video.py
# Videos saved to ./videos/
```

### 5. Plot the training curve
```bash
python plot_rewards.py
```

### 6. Compare PPO vs A2C
```bash
python compare_algorithms.py
```

---

## 📁 Project Structure