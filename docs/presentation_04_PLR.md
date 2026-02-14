# 4. Prioritized Level Replay (Robust PLR)

## Core Concept

**Goal**: Maximize learning by spending training time on levels where agent gains most information.

**Key insight**: Not all training episodes are equally valuable
- **Too easy**: Agent succeeds easily, learns nothing new
- **Too hard**: Agent fails immediately, no learning signal
- **Just right**: Agent struggles but improves ⭐

---

## Algorithm Components

### 1. Level Buffer
Stores visited environment configurations (up to 4000 levels)

Each level tracks:
```python
config: dict              # Environment parameters
score: float              # Learning potential (regret proxy)
visit_count: int          # Times replayed
staleness: int            # Steps since last visit
returns: List[float]      # Episode outcomes
```

---

### 2. Scoring Function

Measures **regret** = how much agent could improve on this level

#### Proxy Metrics:
1. **TD-error**: |Q(s,a) - (r + γ max Q(s',a'))|
   - Large TD-error = bad predictions = room for improvement

2. **Return variance**: Inconsistent outcomes = unpredictable level

#### Score Update (EWA):
```python
level.score = (1 - α) × old_score + α × new_score
```

---

### 3. Replay Decision (Proportionate Schedule)

```python
if buffer.proportion_filled >= ρ and random() < replay_prob:
    level = sample_from_buffer(prioritized=True)
else:
    level = new_random_level()
```

**Default**: `replay_prob = 0.95` (95% replay, 5% novelty)

---

### 4. Prioritized Sampling

#### Rank-based Transform:
```python
# Sort levels by score (descending)
ranks = argsort(scores)[::-1]
weights = ranks^(-1/temperature)  
# High score → low rank → high weight
```

#### Staleness Weighting:
```python
final_weights = (1 - λ) × score_weights 
                + λ × staleness_weights
```

**Result**: High-regret levels sampled more, but stale levels eventually revisited

---

## Implementation: `LevelSampler` Class

```python
sampler = LevelSampler(
    buffer_size=4000,        # Max levels
    replay_prob=0.95,        # 95% replay
    score_transform='rank',  # Rank-based
    temperature=0.1,         # Prioritization strength
    staleness_coef=0.3,      # 30% staleness weight
)
```

### Key Methods:
```python
sampler.add_level(config)                    # Store
sampler.sample_replay_level()                # Sample
sampler.update_score(seed, td_error, return) # Update
```

---

## Training Loop Integration

```python
for episode in training:
    # Decide: replay or new?
    if sampler.sample_replay_decision():
        seed = sampler.sample_replay_level()  # PLR
        config = sampler.levels[seed].config
    else:
        config = random_configuration()  # Exploration
    
    # Run episode
    returns, td_errors = rollout(config)
    
    # Update score
    sampler.update_score(seed, 
                        td_error=mean(td_errors), 
                        episode_return=returns)
```

---

## Robust PLR Enhancements

Standard PLR can fail when buffer fills with impossible levels.

### Our Additions:

1. **Minimum Fill Ratio** (ρ=0.1):
   - Start replay only when buffer 10% full
   - Ensures diverse initial exploration

2. **Staleness-aware Sampling**:
   - Old levels revisited → prevent forgetting
   - Balance exploitation (regret) vs exploration (staleness)

3. **Score Decay** (EWA, α=1.0):
   - Recent performance matters more
   - Difficulty may change as agent improves
