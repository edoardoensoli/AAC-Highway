# 2. Baselines

## Why Baselines Matter

Baselines establish performance benchmarks to measure curriculum learning effectiveness.

---

## 2.1 Deep Q-Network (DQN)

### Algorithm Overview

**DQN** (Mnih et al., 2015) - Value-based RL

**Core components:**
1. **Q-Learning**: Learn action-value function Q(s,a)
2. **Deep Neural Networks**: Approximate Q for large state spaces
3. **Experience Replay**: Break temporal correlations
4. **Target Network**: Stabilize training

```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
                              ↑ target network
```

---

## DQN Implementation: `dqn_baseline.py`

### Architecture
- **Network**: [256, 256] fully-connected
- **Observation**: Kinematics (7 vehicles, 5 features)  
- **Action space**: 5 discrete (IDLE, LEFT, RIGHT, FASTER, SLOWER)

### Hyperparameters
```python
learning_rate: 5e-4
gamma: 0.95                # discount factor
buffer_size: 100k          # replay buffer
batch_size: 64
train_freq: 4              # update every 4 steps
target_update: 250         # sync target network
exploration: ε-greedy (1.0 → 0.05 over 30% training)
```

---

## DQN Training Setup

- **Environment**: highway-fast-v0 (15× speedup)
- **Parallel envs**: 4 SubprocVecEnv (CPU parallelization)
- **Fixed scenario**: 3 lanes, 12 vehicles, 60s episodes
- **Total timesteps**: 500k

### Reward Configuration (ACCEL-aligned)
```python
collision_reward: -10.0    # Strong penalty (not -5.0)
high_speed_reward: 0.3     # Incentivize speed
normalize_reward: False    # Raw rewards
```

---

## 2.2 Proximal Policy Optimization (PPO)

### Algorithm Overview

**PPO** (Schulman et al., 2017) - Policy-based RL

**Core components:**
1. **Policy Gradient**: Directly optimize policy π(a|s)
2. **Clipped Objective**: Prevent destructively large updates
3. **On-policy**: Learn from recent experience
4. **Actor-Critic**: Value network for variance reduction

### Clipped Objective
```
L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]

where r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
```

---

## DQN vs PPO Trade-offs

| Aspect | DQN | PPO |
|--------|-----|-----|
| **Type** | Value-based (Q) | Policy-based |
| **Action space** | Discrete | Discrete/Continuous |
| **Sample efficiency** | High (replay) | Moderate (on-policy) |
| **Stability** | Needs tuning | More robust |
| **Stochastic policy** | No (ε-greedy) | Yes (natural) |
| **Parallelization** | Easy (off-policy) | Essential |

**Our choice**: DQN (simpler, faster, discrete actions)

**Status**: PPO not implemented (future work)

---

## 2.3 Baseline Results (DQN, 500k steps)

### Training Environment (3 lanes, 12 vehicles)
✅ **Survival rate**: ~85%  
✅ **Average reward**: ~2.5-3.0  
✅ **Converges** to local optimum

### Test Environments (Unseen)
- ✅ **Easy** (2 lanes, 8 vehicles): 90% survival
- ⚠️ **Medium** (3 lanes, 20 vehicles): 60% survival
- ❌ **Hard** (4 lanes, 30 vehicles): **20% survival**
- ❌ **Expert** (Aggressive NPCs): **10% survival**

---

## Key Limitations

### 1. Overfitting
- Excellent on training scenario
- Poor on variations
- Agent learns exploitation, not robust skills

### 2. No Exploration of Difficulty Frontier
- All 500k steps on same difficulty
- Cannot learn from harder scenarios
- "Chicken-and-egg" problem

### 3. Sample Inefficiency
- After ~200k steps: 80% survival
- Remaining 300k steps: little new signal
- **Wasted compute on trivial scenarios**

### 4. Lack of Robustness
- High reward ≠ safe driving
- Risky strategies work on training env
- **Catastrophic failures on edge cases**

---

## Motivation for Curriculum Learning

**Key insight**: Agent needs **progressive difficulty**

✅ Build robust skills: simple → complex  
✅ Spend time where it struggles (learning frontier)  
✅ Maintain competence on easy scenarios (prevent forgetting)
