# 3. Curriculum Learning Overview

## The Curriculum Learning Paradigm

**Core idea**: Learn complex tasks by training on a sequence of progressively harder subtasks.

Inspired by human/animal learning:
- 🚼 Babies learn to walk before running
- 🎓 Students learn arithmetic before calculus
- 🚗 Drivers practice in parking lots before highways

---

## How to Create "Levels"

A **level** = environment configuration defining scenario difficulty

### Example: Easy Level (Stage 0)
```python
lanes_count: 2           # Narrow road
vehicles_count: 8        # Sparse traffic
vehicles_density: 0.8    # Comfortable spacing
duration: 30             # Short episode
other_vehicles_type: IDMVehicle  # Predictable
```

### Example: Hard Level (Stage 6)
```python
lanes_count: 4           # Complex navigation
vehicles_count: 30       # Dense traffic
vehicles_density: 1.5    # Tight spacing
duration: 60             # Long survival
other_vehicles_type: AggressiveVehicle  # Unpredictable
```

---

## Difficulty Dimensions

1. **Traffic density**: vehicles_count, vehicles_density
2. **Road complexity**: lanes_count
3. **Episode length**: duration (longer = more failure chances)
4. **NPC behavior**: IDM (safe) → Aggressive (risky)

---

## Curriculum Methods

### Method 1: Fixed Curriculum (Handcrafted)
- Designer manually defines stages: Easy → Medium → Hard
- Agent advances when threshold met (e.g., 80% survival)

**Pros**: Simple, interpretable  
**Cons**: Requires domain expertise, may not be optimal

---

### Method 2: Domain Randomization (DR)
- Sample random configurations from parameter space
- Hope diversity leads to robustness

**Pros**: Simple to implement  
**Cons**: Time wasted on too-easy AND too-hard scenarios

---

### Method 3: Prioritized Level Replay (PLR) ⭐
- Maintain buffer of visited levels
- Prioritize replaying high-"regret" levels (learning potential)

**Pros**: Data-efficient, automatic calibration  
**Cons**: Only replays seen levels, no exploration

---

### Method 4: ACCEL (PLR + Level Editing) ⭐⭐
- PLR + automatic generation via mutations
- Explores frontier of agent capability

**Pros**: Best sample efficiency, adaptive  
**Cons**: More complex implementation

---

## Our Focus

**Methods 3-4**: PLR and ACCEL

Reasons:
- State-of-the-art in curriculum learning research
- Proven effectiveness in Procgen, MiniGrid benchmarks
- Automatic difficulty adaptation (no manual tuning)
