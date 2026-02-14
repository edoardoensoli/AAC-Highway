# 1. Project Introduction

**Edoardo Ensoli 1918623  |  Daniel Munera Martinelli 2049054**

*Adaptive Adversarial Curriculum for Robustness Enhancement in Highway Environment*

---

## Objective
Development of an advanced training framework for **autonomous driving agents** to enhance safety and robustness through **adaptive curriculum learning**.

---

## The Problem

Traditional RL training on fixed environments leads to:
- **Overfitting** to specific traffic patterns
- **Poor generalization** to edge cases  
- **Catastrophic forgetting** when difficulty increases
- **Sample inefficiency** (wasting compute on trivial scenarios)

---

## Our Proposal: Two-Component System

### 1. The Driver Agent (Student)
The autonomous vehicle learning to navigate traffic safely.

### 2. The Traffic Manager (Instructor)
Algorithm that designs progressively challenging scenarios based on agent performance.

---

## The Environment: Highway-Env

**Highway-Env** (Farama Foundation) - Simulation environment for autonomous driving research.

### Key Features:
- **Multi-lane highway** with dynamic traffic
- **Kinematic observations**: relative positions, velocities of nearby vehicles
- **Discrete actions**: lane changes (LEFT/RIGHT), speed control (FASTER/SLOWER), IDLE
- **Realistic physics**: IDM (Intelligent Driver Model) for NPC vehicles

---

## Configurable Parameters

```python
lanes_count: 2-4           # Road width
vehicles_count: 5-50       # Traffic density (absolute)
vehicles_density: 0.5-2.0  # Relative spacing
duration: 30-80 seconds    # Episode length
other_vehicles_type:       # IDM (predictable) 
                           # vs Aggressive (unpredictable)
```

---

## Observation Space

**Kinematics**: 7 vehicles × 5 features
- `presence`: Is vehicle slot occupied?
- `x, y`: Relative position
- `vx, vy`: Relative velocity

**Properties**:
- Normalized to [-1, 1]
- Sorted by distance
- See behind (rear-view mirror)

Shape: `(7, 5)` tensor

---

## Reward Structure

```python
reward = high_speed_reward × speed_fraction 
         - collision_penalty × crashed
```

- **Speed reward**: +0.15 to +0.3 per step (25-30 m/s)
- **Collision penalty**: -10.0 (terminal)
- **Raw rewards** (no normalization) for clearer learning signal

**Design goal**: Encourage fast but **safe** driving
