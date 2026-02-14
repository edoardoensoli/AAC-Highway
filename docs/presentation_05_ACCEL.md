# 5. ACCEL (Evolving Curricula with Regret-Based Environment Design)

## Core Concept

**PLR limitation**: Only replays *seen* levels, cannot explore new difficulty frontiers

**ACCEL innovation**: Automatically **generate new levels** by mutating high-regret levels

**Metaphor**: 
- PLR = study existing hard problems
- ACCEL = create new hard problems tailored to your weaknesses

---

## Algorithm: PLR + Level Editor

### Level Editing (Mutations)

**Operation**:
1. Select high-regret level from buffer
2. Randomly perturb 1-3 parameters by small amounts
3. Add mutated level to buffer

**Example**:
```python
# Parent level
base = {lanes: 3, vehicles: 15, density: 1.0}

# Mutation: vehicles_count +5
mutated = {lanes: 3, vehicles: 20, density: 1.0}
```

---

## Mutation Space

```python
PARAM_SPACE = {
    'vehicles_count':  (5, 50),    step=5
    'vehicles_density': (0.5, 2.0), step=0.2
    'lanes_count':     (2, 4),     step=1
    'duration':        (30, 80),   step=10
}
```

### Bounded Mutations (Key for Stability)
- Mutations restricted to **current stage bounds**
- Prevents "chain drift" to impossible difficulty
- Example: Stage 2 mutations stay within Stage 2-3 difficulty

---

## Our Implementation: Fixed Stages + Adaptive Exploration

Combines:
1. **Fixed difficulty stages**: 8 hand-designed anchors
2. **ACCEL mutations**: Explore *around* current stage

**Why hybrid?**
- Fixed stages: Ensure progression coverage
- Mutations: Discover edge cases dynamically

---

## 8-Stage Curriculum

```
Stage 0: 2 lanes, 8 veh, IDM, 30s       # Basic avoidance
Stage 1: 3 lanes, 8 veh, IDM, 30s       # Multi-lane
Stage 2: 3 lanes, 15 veh, IDM, 30s      # Moderate traffic
Stage 3: 3 lanes, 15 veh, AGGR, 30s     # Unpredictability ⚠️
Stage 4: 3 lanes, 15 veh, Aggr, ρ↑, 30s
Stage 5: 3 lanes, 15 veh, Aggr, ρ↑, 50s # Long-term
Stage 6: 3 lanes, 20 veh, Aggr, 50s
Stage 7: 4 lanes, 30 veh, Aggr, ρ↑, 60s # Final
```

**Design principle**: Change **ONE variable at a time**

---

## Key Innovation: Early Aggressive Vehicles

**Stage 3**: Introduce aggressive vehicles on *moderate traffic*

**Why?**
- Previous designs: aggressive at Stage 5 (after 5 stages of predictable behavior)
- Problem: Agent overfits to predictable IDM, difficult to adapt
- Solution: Introduce unpredictability early, before dense traffic

**Result**: Learn defensive driving on manageable difficulty first

---

## Mastery-Based Progression

### Advancement Criteria:
```python
# Continuous survival metric (not binary)
survival = avg(episode_length / expected_length) 
           over last 50 eval episodes

if survival >= 0.80:  # 80% threshold
    advance_to_next_stage()
```

### Requirements:
- **Minimum 50 evaluation episodes** on pure stage config
- **80% average survival** (continuous metric)
- Prevents premature advancement

---

## Stage Rehearsal (Anti-Forgetting)

**Problem**: As agent advances, may forget easy scenarios

**Solution**: 10% of episodes sample from **previous mastered stages**

```python
if len(mastered_stages) > 0 and random() < 0.10:
    old_stage = random_choice(mastered_stages)
    config = get_stage_config(old_stage)
```

**Result**: Maintains competence on all difficulty levels

---

## Competence Gate (Curriculum Safeguard)

### Problem: PLR Death Spiral
1. Agent fails on mutated levels → high regret
2. PLR replays those levels more → agent keeps failing
3. No improvement, training stalls ❌

### Solution: Competence-based Control

```python
if survival < 50%:
    # Phase 1: ONLY stage config, NO PLR/ACCEL
    train_on_pure_config()
    
elif survival < 75%:
    # Phase 2: Gradual PLR activation
    plr_prob = scale(survival, 50% → 75%)
    
else:
    # Phase 3: Full PLR/ACCEL
    plr_prob = 0.95
```

**Rationale**: Master current stage before exploring variations

---

## Curriculum Retention (Anti-Catastrophic-Forgetting)

### Problem Discovered at 1M Steps
- Buffer fills with new difficult levels
- Old easy levels **removed** (lowest score)
- Agent **forgets how to brake** ⚠️

### Solution: Core Level Protection

#### 1. Mark Levels as "Core"
```python
# When advancing Stage 2 → Stage 3
mark_best_5_levels(stage=2, is_core=True)
```

#### 2. Protected Removal
```python
if buffer_full:
    # Remove ONLY non-core with low score
    removable = [l for l in buffer if not l.is_core]
    remove(min(removable, key=score))
```

**Result**: Buffer always contains representative levels from all mastered stages

---

## Training Flow (Per-Episode Decision)

```python
def decide_next_level(env_idx):
    # 1. Evaluation (30%): pure stage config
    if random() < 0.3:
        return current_stage_config()
    
    # 2. Stage rehearsal (10%): old stages
    if random() < 0.10 and len(mastered) > 0:
        return sample_old_stage()
    
    # 3. Competence gate check
    if survival < 0.50:
        return current_stage_config()  # NO mutations
    
    # 4. PLR replay (95% of remaining)
    if random() < 0.95:
        return sampler.sample_replay_level()
    
    # 5. ACCEL mutation (5%)
    parent = sampler.sample_replay_level()
    return generator.mutate_level(parent)
```

---

## ACCEL vs PLR Comparison

| Feature | PLR | ACCEL |
|---------|-----|-------|
| **Level source** | Fixed pool | Fixed + generated |
| **Exploration** | Finite | Infinite (mutations) |
| **Sample efficiency** | High | Highest |
| **Difficulty frontier** | Slow | Actively seeks |
| **Robustness** | Good | Better (diversity) |
| **Complexity** | Moderate | High |
| **Forgetting risk** | Low | Mitigated (core + rehearsal) |

---

## When to Use Each?

**PLR**: Limited compute, simple domain, known difficulty range

**ACCEL**: Complex domain, need maximum robustness, sufficient compute
