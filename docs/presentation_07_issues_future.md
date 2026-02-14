# 7. Issues and Limitations

## SOTA Performance Requires MUCH More Training

### State-of-the-Art (Literature)
- **10-50M timesteps** for highway-env mastery
- Research labs: 64-128 parallel envs on clusters
- Training time: 10M steps in 6-12 hours

### Our Training
- **2M timesteps** (4-5% of SOTA)
- 4 parallel envs on consumer laptop
- Training time: 10M steps would take **5-7 days** ⚠️

**Impact**: Proof-of-concept demonstrated, but not absolute SOTA performance

---

## Hardware Limitations: CPU-Bounded

### Training Bottleneck
**Simulation speed**, not model updates

### Our Setup
- No GPU training (DQN works on CPU)
- 4 parallel environments
- highway-fast-v0 (15× speedup, but still slow)

### Training Times
```
200k steps: ~2-3 hours
500k steps: ~6-8 hours
1M steps:   ~12-15 hours
2M steps:   ~24-30 hours (1 full day)
```

### Why Slow?
- Highway-env simulates vehicle physics, collision detection
- Python GIL limits parallelism
- SubprocVecEnv overhead (IPC between processes)

---

## Other Limitations

### Exploration-Exploitation Trade-off
- ACCEL mutations sometimes create *too hard* levels
- Competence gate mitigates but doesn't eliminate
- Could benefit from **adaptive mutation magnitude**

### Hyperparameter Sensitivity
- Mastery threshold (80%): too high → stuck, too low → premature advancement
- Stage rehearsal (10%): balance retention vs progress
- **Limited tuning** due to compute constraints

### Evaluation Variance
- Highway-env stochastic (random NPC spawns)
- 10 episodes per scenario **insufficient for statistical confidence**
- Should use 50-100 episodes (but expensive)

### Missing PPO Baseline
- PPO likely better for continuous control
- No direct comparison (time constraints)

---

# 8. Future Work

## Short-Term Improvements

### 1. Complete PPO Implementation
**Why?**
- More stable training
- Better exploration (stochastic policy)
- Possibly faster convergence

**Expected benefits**:
- Direct comparison DQN vs PPO
- Better baseline for curriculum evaluation

---

### 2. Extended Training
**Goal**: Train ACCEL to **5-10M steps** on better hardware

**Expected results**:
- SOTA performance (90%+ expert survival)
- Compare to published results

**Requirements**:
- Access to GPU cluster or cloud compute
- 64+ parallel environments

---

### 3. Hyperparameter Optimization

**Grid search**:
- Mastery threshold: 0.70-0.90
- Stage rehearsal: 5%, 10%, 15%, 20%
- Mutation magnitude: adaptive based on competence
- PLR replay prob: 0.90-0.98

**Expected**: 10-20% performance improvement

---

### 4. More Diverse Test Scenarios

**Add**:
- Merging lanes
- Roundabouts (if highway-env supports)
- Sudden braking, cut-in maneuvers

**Transfer learning test**:
- Train on highway
- Test on urban roads

---

## Research Extensions

### 1. Adversarial Traffic Manager ⭐ (Original Proposal)

**Current**: ACCEL mutations are **random perturbations**

**Proposal**: Train adversarial agent to **design difficult scenarios**

#### Algorithm:
```
Traffic Manager = RL agent
Reward = Driver's regret (how much Driver struggles)
Co-evolution: Driver vs Manager min-max game
```

#### Expected Benefits:
- ✅ More targeted difficulty increases
- ✅ Automatically discovers edge cases (e.g., "cut-in trap")
- ✅ Learns which parameter combinations challenge agent most

#### Challenges:
- ❌ **Sparse rewards**: Manager only gets feedback after full episode
- ❌ **Credit assignment**: Which parameter change caused failure?
- ❌ **Sample efficiency**: Need 2× training (Driver + Manager)

---

#### Implementation Approach

**Phase 1**: Train Driver with ACCEL (baseline)
- 2M steps, master Stage 0-7
- Save checkpoints

**Phase 2**: Freeze Driver, train Manager
- Manager observes Driver's policy
- Generates configs, measures Driver's TD-error
- Reward = max(TD-error) over episode

**Phase 3**: Alternating co-evolution
- Train Driver on Manager's hard scenarios
- Train Manager to challenge improved Driver
- Iterate until convergence

**Expected timeline**: 5-10M steps total

---

### 2. Hierarchical Curriculum

**Current**: 8 fixed stages

**Proposal**: Multi-granularity curriculum
- **Macro-level**: Easy/Medium/Hard (as now)
- **Micro-level**: Within each, ACCEL mutations
- **Meta-level**: Learn *when* to advance stages (adaptive threshold)

**Benefits**:
- Smoother difficulty progression
- Automatic stage discovery (no hand-design)

---

### 3. Multi-Agent Curriculum

**Current**: Single ego vehicle training

**Proposal**: Train multiple agents simultaneously
- **Easy**: Agent competes with poor drivers
- **Hard**: Agent competes with other RL agents

**Benefits**:
- More realistic NPC behaviors (human-like diversity)
- Emergent scenarios (multi-agent interactions)

**Example**: Agent learns to predict/react to aggressive learner agents

---

### 4. Transfer Learning: Sim-to-Real

**Ultimate goal**: Apply learned policy to **real autonomous vehicles**

#### Approach:
1. **Domain randomization**: Vary observation noise, dynamics
2. **Observation adaptation**: RGB cameras → kinematic state
3. **Safety wrapper**: Rule-based emergency braking on top of RL
4. **Gradual real-world deployment**:
   - Test track (controlled)
   - Highway with safety driver
   - Full autonomy (if regulations permit)

#### Challenges:
- ❌ **Reality gap**: Simulation never perfectly models real world
- ❌ **Safety criticality**: Cannot afford catastrophic failures
- ❌ **Regulatory**: Autonomous vehicle testing requires permits

---

## Engineering Improvements

### 1. Distributed Training
- Scale to 64+ parallel envs on cluster
- Use **Ray RLlib** or similar framework
- Target: **10M steps in <12 hours**

### 2. Better Logging/Visualization
- Real-time TensorBoard: survival per stage, buffer heatmap
- **Episode recordings**: video of best/worst episodes
- Curriculum visualization: stage visits over time

### 3. Adaptive Simulation Speed
- Use highway-fast-v0 (15×) during early training
- Switch to highway-v0 (1×) for final evaluation (more accurate)

### 4. Model Architecture Experiments
- Current: [256, 256] MLP
- Try: **Recurrent (LSTM/GRU)** for temporal dependencies
- Try: **Attention mechanism** for variable vehicles

---

# 9. References

## Core Papers

**DQN:**
- Mnih et al. (2015). *Human-level control through deep RL*. Nature.
  - [https://www.nature.com/articles/nature14236](https://www.nature.com/articles/nature14236)

**PPO:**
- Schulman et al. (2017). *Proximal Policy Optimization*. arXiv:1707.06347.
  - [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347)

**PLR:**
- Jiang et al. (2021). *Prioritized Level Replay*. ICML 2021.
  - [https://arxiv.org/abs/2010.03934](https://arxiv.org/abs/2010.03934)

**ACCEL:**
- Parker-Holder et al. (2022). *Evolving Curricula with Regret-Based Environment Design*. ICML 2022.
  - [https://arxiv.org/abs/2203.01302](https://arxiv.org/abs/2203.01302)
  - [https://accelagent.github.io/](https://accelagent.github.io/)

---

## Environment and Tools

**Highway-Env:**
- Leurent (2018). *An Environment for Autonomous Driving Decision-Making*.
  - [https://github.com/Farama-Foundation/HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)

**Stable-Baselines3:**
- Raffin, Hill et al. (2021). *Stable-Baselines3*.
  - [https://github.com/DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3)

---

## Related Work

**Domain Randomization:**
- Tobin et al. (2017). *Domain Randomization for Sim-to-Real Transfer*. IROS.
  - [https://arxiv.org/abs/1703.06907](https://arxiv.org/abs/1703.06907)

**Automatic Curriculum:**
- Graves et al. (2017). *Automated Curriculum Learning*. ICML.
  - [https://arxiv.org/abs/1704.03003](https://arxiv.org/abs/1704.03003)

**Autonomous Driving RL:**
- Bojarski et al. (2016). *End-to-End Learning for Self-Driving Cars*. NVIDIA.
  - [https://arxiv.org/abs/1604.07316](https://arxiv.org/abs/1604.07316)

---

# Thank You!

**Questions?**

---

**Edoardo Ensoli** 1918623  
**Daniel Munera Martinelli** 2049054

**GitHub**: [Your repository link]
