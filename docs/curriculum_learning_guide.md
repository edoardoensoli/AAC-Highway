# Comprehensive Guide to Curriculum Learning for RL Generalization

## Overview
This guide covers three progressive approaches for training RL agents that generalize well to environment changes:

1. **Domain Randomization (DR)** - Simplest, randomizes environment parameters
2. **Prioritized Level Replay (PLR)** - Intelligently samples environments based on learning potential
3. **ACCEL** - Evolves curriculum by mutating high-regret environments

---

## 1. Domain Randomization (DR)

### Theory

**Core Idea**: Train on a diverse set of randomly sampled environment parameters so the agent learns robust policies that treat the real world as "just another variation."

**Key Principle**: If training distribution is wide enough, the test environment falls within the distribution, enabling zero-shot transfer.

**Mathematics**:
- Training distribution: θ ~ P(θ) where θ are environment parameters
- Goal: Learn policy π that maximizes E_θ~P(θ)[R(π, θ)]
- Assumes test environment θ_test is covered by P(θ)

**Advantages**:
- Simple to implement
- No curriculum learning complexity
- Proven effective for sim2real transfer

**Disadvantages**:
- Uniform sampling can be inefficient
- May spend too much time on easy/irrelevant variations
- No adaptive focus on challenging scenarios

### Key Papers

1. **"Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"**
   - Authors: Josh Tobin et al.
   - Year: 2017
   - arXiv: https://arxiv.org/abs/1703.06907
   - First major DR paper for robotics, showed object detection transfer

2. **"Sim-to-Real Transfer of Robotic Control with Dynamics Randomization"**
   - Authors: Xue Bin Peng et al.
   - Year: 2018
   - arXiv: https://arxiv.org/abs/1710.06537
   - Dynamics randomization for locomotion

3. **"Learning Dexterous In-Hand Manipulation"**
   - Authors: OpenAI et al.
   - Year: 2018
   - arXiv: https://arxiv.org/abs/1808.00177
   - Rubik's cube solving with extensive DR

4. **"Understanding Domain Randomization for Sim-to-Real Transfer"**
   - Authors: Xiong et al.
   - Year: 2022
   - Conference: ICLR 2022
   - URL: https://openreview.net/pdf?id=T8vZHIRTrY
   - Theoretical analysis of when/why DR works

### Implementation Details

#### Basic DR Implementation

```python
import numpy as np
import gymnasium as gym

class DomainRandomization:
    """Simple domain randomization for highway-env"""
    
    def __init__(self, env_id="highway-v0"):
        self.env_id = env_id
        self.param_ranges = {
            'lanes_count': [2, 3, 4, 5],
            'vehicles_count': (15, 40),
            'vehicles_density': (0.8, 2.5),
            'duration': (40, 80),
            'collision_reward': (-5.0, -0.5),
            'high_speed_reward': (0.2, 0.8),
        }
    
    def sample_config(self):
        """Sample random environment configuration"""
        config = {}
        
        # Discrete parameters
        config['lanes_count'] = np.random.choice(self.param_ranges['lanes_count'])
        
        # Continuous parameters
        config['vehicles_count'] = int(np.random.uniform(*self.param_ranges['vehicles_count']))
        config['vehicles_density'] = np.random.uniform(*self.param_ranges['vehicles_density'])
        config['duration'] = int(np.random.uniform(*self.param_ranges['duration']))
        
        # Reward shaping randomization
        config['collision_reward'] = np.random.uniform(*self.param_ranges['collision_reward'])
        config['high_speed_reward'] = np.random.uniform(*self.param_ranges['high_speed_reward'])
        
        return config
    
    def create_env(self, config=None):
        """Create environment with given or random config"""
        if config is None:
            config = self.sample_config()
        
        env = gym.make(self.env_id, config=config, render_mode='rgb_array')
        return env

# Usage with Stable Baselines3
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

def make_randomized_env():
    """Factory function for parallel environments"""
    dr = DomainRandomization()
    return lambda: dr.create_env()

# Create vectorized environment with DR
env = SubprocVecEnv([make_randomized_env() for _ in range(8)])

model = DQN('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=100000)
```

#### Advanced: Automatic Domain Randomization (ADR)

```python
class AutomaticDomainRandomization:
    """
    ADR gradually expands randomization ranges based on performance
    Similar to OpenAI's approach for Rubik's cube
    """
    
    def __init__(self, initial_ranges, expansion_rate=0.05):
        self.ranges = initial_ranges.copy()
        self.initial_ranges = initial_ranges.copy()
        self.expansion_rate = expansion_rate
        self.performance_buffer = []
        
    def sample_config(self):
        config = {}
        for param, (min_val, max_val) in self.ranges.items():
            if isinstance(min_val, int):
                config[param] = np.random.randint(min_val, max_val + 1)
            else:
                config[param] = np.random.uniform(min_val, max_val)
        return config
    
    def update_ranges(self, success_rate):
        """Expand ranges if agent is doing well"""
        self.performance_buffer.append(success_rate)
        
        if len(self.performance_buffer) < 100:
            return
        
        avg_performance = np.mean(self.performance_buffer[-100:])
        
        # If performing well, expand randomization
        if avg_performance > 0.7:
            for param, (min_val, max_val) in self.ranges.items():
                initial_min, initial_max = self.initial_ranges[param]
                
                # Expand ranges
                range_size = max_val - min_val
                expansion = range_size * self.expansion_rate
                
                new_min = max(min_val - expansion, initial_min * 0.5)
                new_max = min(max_val + expansion, initial_max * 1.5)
                
                self.ranges[param] = (new_min, new_max)
```

---

## 2. Prioritized Level Replay (PLR)

### Theory

**Core Idea**: Not all environment configurations are equally useful for learning. PLR prioritizes training on environments where the agent has high "learning potential."

**Learning Potential**: Measured by TD-error or value loss - high error indicates the agent is still learning from that environment.

**Mathematics**:
- For each level/environment l, maintain score S(l)
- Score based on TD-error: S(l) = |V(s) - (r + γV(s'))|
- Sample next level with probability ∝ S(l) + staleness(l)
- Staleness encourages revisiting environments not seen recently

**Key Innovation**: Adaptive curriculum emerges automatically - agent naturally progresses from easy to hard levels as it masters each.

**Staleness Correction**: Prevents score drift by adding bonus for levels not seen recently:
```
sampling_weight(l) = score(l) + staleness_weight * timesteps_since_visited(l)
```

**Advantages**:
- Efficient sample use - focuses on useful environments
- Automatic curriculum without manual design
- Proven superior to uniform sampling
- Works with any RL algorithm

**Disadvantages**:
- Requires maintaining scores for all seen levels
- Limited to existing levels (doesn't generate new ones)
- Can overfit to specific level patterns

### Key Papers

1. **"Prioritized Level Replay" (Original Paper)**
   - Authors: Minqi Jiang, Edward Grefenstette, Tim Rocktäschel
   - Conference: ICML 2021
   - arXiv: https://arxiv.org/abs/2010.03934
   - Code: https://github.com/facebookresearch/level-replay
   - Core PLR algorithm with TD-error scoring

2. **"Replay-Guided Adversarial Environment Design"**
   - Authors: Jiang et al.
   - Conference: NeurIPS 2021
   - Paper: https://proceedings.neurips.cc/paper/2021/hash/0e915db6326b6fb6a3c56546980a8c93-Abstract.html
   - Combines PLR with adversarial environment generation (PAIRED)
   - Introduces PLR⊥ with theoretical guarantees

3. **"Emergent Complexity and Zero-Shot Transfer via Unsupervised Environment Design"**
   - Authors: Dennis et al.
   - Year: 2020
   - Introduces PAIRED algorithm (complements PLR)

### Implementation Details

```python
import numpy as np
from collections import defaultdict, deque
import torch

class PrioritizedLevelReplay:
    """
    PLR implementation for curriculum learning
    Based on Jiang et al. 2021
    """
    
    def __init__(
        self,
        env_configs,
        score_function='value_loss',
        temperature=0.1,
        staleness_coef=0.1,
        buffer_size=100,
        replay_prob=0.8,  # Probability of replaying vs sampling new
    ):
        self.env_configs = env_configs
        self.score_function = score_function
        self.temperature = temperature
        self.staleness_coef = staleness_coef
        self.buffer_size = buffer_size
        self.replay_prob = replay_prob
        
        # Tracking
        self.scores = defaultdict(lambda: 1.0)  # Initial score
        self.staleness = defaultdict(int)
        self.visit_count = defaultdict(int)
        self.seen_levels = set()
        self.level_buffer = []  # Top-K levels
        
        self.timestep = 0
        
    def compute_score(self, level_id, advantages, values, returns):
        """
        Compute learning potential score for a level
        
        Args:
            level_id: Environment identifier
            advantages: GAE advantages from the episode
            values: Value predictions
            returns: Actual returns
        """
        if self.score_function == 'value_loss':
            # L1 value loss (most common in PLR)
            score = np.mean(np.abs(values - returns))
        elif self.score_function == 'max_mc':
            # Maximum Monte Carlo return
            score = np.max(returns)
        elif self.score_function == 'gae':
            # Mean absolute GAE advantage
            score = np.mean(np.abs(advantages))
        elif self.score_function == 'positive_advantage':
            # Mean of positive advantages only
            positive_adv = advantages[advantages > 0]
            score = np.mean(positive_adv) if len(positive_adv) > 0 else 0.0
        else:
            raise ValueError(f"Unknown score function: {self.score_function}")
        
        return float(score)
    
    def update_level_score(self, level_id, score):
        """Update score for a level after training episode"""
        # Exponential moving average
        alpha = 0.1
        self.scores[level_id] = alpha * score + (1 - alpha) * self.scores[level_id]
        
        # Update buffer if needed
        self._update_buffer(level_id)
    
    def _update_buffer(self, level_id):
        """Maintain top-K levels by score"""
        if level_id not in self.level_buffer:
            self.level_buffer.append(level_id)
        
        # Sort and trim to buffer size
        self.level_buffer.sort(key=lambda l: self.scores[l], reverse=True)
        self.level_buffer = self.level_buffer[:self.buffer_size]
    
    def sample_level(self):
        """
        Sample next training level using PLR strategy
        
        Returns:
            (level_id, config): Selected level and its configuration
        """
        self.timestep += 1
        
        # Decide: replay from buffer or sample new level
        if len(self.level_buffer) > 0 and np.random.random() < self.replay_prob:
            # Sample from buffer using prioritization
            level_id = self._sample_from_buffer()
        else:
            # Sample new unseen level
            level_id = self._sample_new_level()
        
        # Update staleness
        for lid in self.level_buffer:
            self.staleness[lid] += 1
        self.staleness[level_id] = 0
        
        # Track visit
        self.visit_count[level_id] += 1
        self.seen_levels.add(level_id)
        
        return level_id, self.env_configs[level_id]
    
    def _sample_from_buffer(self):
        """Sample level from buffer using score + staleness"""
        weights = []
        for level_id in self.level_buffer:
            # Combine score with staleness bonus
            w = self.scores[level_id] + self.staleness_coef * self.staleness[level_id]
            weights.append(w)
        
        weights = np.array(weights)
        
        # Temperature-based softmax
        if self.temperature > 0:
            probs = np.exp(weights / self.temperature)
            probs = probs / probs.sum()
        else:
            # Greedy selection
            probs = np.zeros(len(weights))
            probs[np.argmax(weights)] = 1.0
        
        idx = np.random.choice(len(self.level_buffer), p=probs)
        return self.level_buffer[idx]
    
    def _sample_new_level(self):
        """Sample a new unseen level uniformly"""
        unseen = [i for i in range(len(self.env_configs)) if i not in self.seen_levels]
        
        if len(unseen) > 0:
            return np.random.choice(unseen)
        else:
            # All levels seen, sample uniformly from all
            return np.random.choice(len(self.env_configs))
    
    def get_stats(self):
        """Get statistics for logging"""
        return {
            'num_seen_levels': len(self.seen_levels),
            'buffer_size': len(self.level_buffer),
            'avg_score': np.mean(list(self.scores.values())) if self.scores else 0,
            'max_score': max(self.scores.values()) if self.scores else 0,
        }


# Integration with Stable Baselines3 DQN
class PLRCallback:
    """Callback to integrate PLR with SB3 training"""
    
    def __init__(self, plr, model, env):
        self.plr = plr
        self.model = model
        self.env = env
        self.current_level_id = None
        self.episode_advantages = []
        self.episode_values = []
        self.episode_returns = []
        
    def on_episode_start(self):
        """Called at start of episode"""
        # Sample new level
        self.current_level_id, config = self.plr.sample_level()
        
        # Update environment with new config
        self.env = gymnasium.make(
            "highway-v0",
            config=config,
            render_mode='rgb_array'
        )
        self.model.set_env(self.env)
        
        # Reset tracking
        self.episode_advantages = []
        self.episode_values = []
        self.episode_returns = []
    
    def on_step(self, obs, action, reward, next_obs, done):
        """Called after each step"""
        # Compute value predictions
        with torch.no_grad():
            value = self.model.policy.predict_values(
                torch.FloatTensor(obs).unsqueeze(0)
            )[0].item()
            
            next_value = self.model.policy.predict_values(
                torch.FloatTensor(next_obs).unsqueeze(0)
            )[0].item()
        
        # Compute TD error (advantage estimate)
        advantage = reward + self.model.gamma * next_value * (1 - done) - value
        
        self.episode_values.append(value)
        self.episode_advantages.append(advantage)
    
    def on_episode_end(self, episode_rewards):
        """Called at end of episode"""
        # Compute returns
        returns = []
        R = 0
        for r in reversed(episode_rewards):
            R = r + self.model.gamma * R
            returns.insert(0, R)
        
        # Update PLR with episode data
        score = self.plr.compute_score(
            self.current_level_id,
            np.array(self.episode_advantages),
            np.array(self.episode_values),
            np.array(returns)
        )
        
        self.plr.update_level_score(self.current_level_id, score)


# Example usage
def train_with_plr():
    """Complete training loop with PLR"""
    # Define environment variations
    env_configs = []
    for lanes in [2, 3, 4]:
        for density in [0.8, 1.0, 1.5, 2.0]:
            for vehicles in [15, 20, 25, 30, 35]:
                config = {
                    'lanes_count': lanes,
                    'vehicles_density': density,
                    'vehicles_count': vehicles,
                    'duration': 60,
                }
                env_configs.append(config)
    
    # Initialize PLR
    plr = PrioritizedLevelReplay(
        env_configs,
        score_function='value_loss',
        temperature=0.1,
        replay_prob=0.8
    )
    
    # Start with first level
    level_id, config = plr.sample_level()
    env = gymnasium.make("highway-v0", config=config, render_mode='rgb_array')
    
    # Create model
    model = DQN('MlpPolicy', env, verbose=1)
    
    # Training loop
    total_timesteps = 100000
    timesteps_per_level = 2048
    
    timesteps = 0
    while timesteps < total_timesteps:
        # Sample level
        level_id, config = plr.sample_level()
        env = gymnasium.make("highway-v0", config=config, render_mode='rgb_array')
        model.set_env(env)
        
        # Train on this level
        model.learn(timesteps_per_level, reset_num_timesteps=False)
        
        # Get value loss from replay buffer for scoring
        # (simplified - in practice extract from model's logger)
        value_loss = model.logger.name_to_value.get('train/value_loss', 1.0)
        
        plr.update_level_score(level_id, value_loss)
        
        timesteps += timesteps_per_level
        
        # Log stats
        if timesteps % 10000 == 0:
            print(f"Timesteps: {timesteps}, PLR stats: {plr.get_stats()}")
    
    return model, plr
```

---

## 3. ACCEL (Adversarially Compounding Complexity by Editing Levels)

### Theory

**Core Idea**: Combine PLR's regret-based prioritization with evolutionary algorithms to *generate* new challenging environments by mutating existing ones.

**Key Innovation**: Don't just sample from fixed levels - evolve new levels at the frontier of agent capabilities.

**Regret-Based Scoring**: 
```
Regret(level) = Return(optimal_policy, level) - Return(current_policy, level)
```
High regret = environment where agent could do much better = high learning potential

**Evolution Process**:
1. Maintain buffer of high-regret levels
2. Sample level proportional to regret
3. Mutate level (add obstacles, change layout, etc.)
4. Evaluate mutated level
5. Add to buffer if high regret

**Mathematics**:
- Level space: L
- Mutation operator: M: L → L (e.g., add/remove obstacles)
- Regret: R(l, π) = V*(l) - V^π(l)
- Sample level: l ~ P(l | R)
- Mutate: l' = M(l)
- Curriculum evolves: L_0 → L_1 → L_2 → ...

**Advantages**:
- Open-ended curriculum generation
- Automatically increases complexity
- Better generalization than PLR
- Proven superior on complex domains

**Disadvantages**:
- Requires level mutation operators (domain-specific)
- More complex to implement
- Computationally expensive
- Need to approximate regret (since optimal policy unknown)

### Key Papers

1. **"Evolving Curricula with Regret-Based Environment Design" (ACCEL)**
   - Authors: Jack Parker-Holder et al.
   - Conference: ICML 2022
   - arXiv: https://arxiv.org/abs/2203.01302
   - Website: https://accelagent.github.io/
   - Main ACCEL paper with full algorithm

2. **"Prioritized Level Replay" (Foundation)**
   - See PLR section
   - ACCEL builds on PLR's regret-based framework

3. **"Emergent Complexity via Multi-Agent Competition" (PAIRED)**
   - Authors: Dennis et al.
   - Conference: ICLR 2021
   - Related adversarial environment design approach

4. **Review: "ACCEL: Evolving Curricula with Regret-Based Environment Design" by Yannic Kilcher**
   - YouTube: Search "Yannic Kilcher ACCEL"
   - Excellent explanation of the algorithm

### Implementation Details

```python
import numpy as np
import copy
from collections import deque

class ACCELCurriculum:
    """
    ACCEL: Adversarially Compounding Complexity by Editing Levels
    Based on Parker-Holder et al. 2022
    """
    
    def __init__(
        self,
        initial_level,
        mutation_fn,
        buffer_size=100,
        regret_estimator='advantage',
        mutation_rate=0.1,
        temperature=0.1,
    ):
        """
        Args:
            initial_level: Starting simple level configuration
            mutation_fn: Function that mutates a level: fn(level) -> new_level
            buffer_size: Size of replay buffer for high-regret levels
            regret_estimator: How to estimate regret ('advantage', 'value_loss', 'returns')
            mutation_rate: Probability of mutation when editing
            temperature: Temperature for sampling from buffer
        """
        self.mutation_fn = mutation_fn
        self.buffer_size = buffer_size
        self.regret_estimator = regret_estimator
        self.mutation_rate = mutation_rate
        self.temperature = temperature
        
        # Initialize buffer with simple level
        self.level_buffer = []
        self.regret_scores = {}
        self.level_counter = 0
        
        # Add initial level
        self._add_level(initial_level, regret=1.0)
        
    def _add_level(self, level, regret):
        """Add level to buffer"""
        level_id = self.level_counter
        self.level_counter += 1
        
        self.level_buffer.append(level_id)
        self.regret_scores[level_id] = regret
        
        # Store level configuration
        if not hasattr(self, 'levels'):
            self.levels = {}
        self.levels[level_id] = copy.deepcopy(level)
        
        # Maintain buffer size
        if len(self.level_buffer) > self.buffer_size:
            # Remove level with lowest regret
            min_id = min(self.level_buffer, key=lambda l: self.regret_scores[l])
            self.level_buffer.remove(min_id)
            del self.regret_scores[min_id]
            del self.levels[min_id]
        
        return level_id
    
    def sample_and_mutate(self):
        """
        ACCEL's main operation: sample high-regret level and mutate it
        
        Returns:
            (level_id, level_config)
        """
        # Sample level proportional to regret
        level_id = self._sample_by_regret()
        parent_level = self.levels[level_id]
        
        # Mutate level
        new_level = self.mutation_fn(parent_level, self.mutation_rate)
        
        # Add mutated level to buffer (with inherited regret initially)
        new_id = self._add_level(new_level, self.regret_scores[level_id])
        
        return new_id, new_level
    
    def _sample_by_regret(self):
        """Sample level proportional to regret score"""
        regrets = np.array([self.regret_scores[l] for l in self.level_buffer])
        
        # Temperature-based sampling
        if self.temperature > 0:
            weights = np.exp(regrets / self.temperature)
            probs = weights / weights.sum()
        else:
            # Greedy
            probs = np.zeros(len(regrets))
            probs[np.argmax(regrets)] = 1.0
        
        idx = np.random.choice(len(self.level_buffer), p=probs)
        return self.level_buffer[idx]
    
    def update_regret(self, level_id, advantages, values, returns):
        """
        Update regret estimate for a level after episode
        
        Regret approximation:
        - True regret: R = V*(s) - V^π(s)
        - We approximate V* using various proxies
        """
        if self.regret_estimator == 'advantage':
            # High positive advantages indicate room for improvement
            regret = np.mean(np.maximum(advantages, 0))
        
        elif self.regret_estimator == 'value_loss':
            # Large value errors indicate misestimation
            regret = np.mean(np.abs(values - returns))
        
        elif self.regret_estimator == 'returns':
            # Low returns indicate difficulty
            # Invert so high regret = hard level
            max_possible_return = 100  # Domain-specific
            regret = max_possible_return - np.mean(returns)
        
        elif self.regret_estimator == 'max_advantage':
            # Maximum advantage seen
            regret = np.max(advantages) if len(advantages) > 0 else 0
        
        else:
            raise ValueError(f"Unknown regret estimator: {self.regret_estimator}")
        
        # Exponential moving average
        alpha = 0.1
        if level_id in self.regret_scores:
            self.regret_scores[level_id] = \
                alpha * regret + (1 - alpha) * self.regret_scores[level_id]
        else:
            self.regret_scores[level_id] = regret
    
    def get_stats(self):
        """Statistics for logging"""
        if len(self.regret_scores) == 0:
            return {}
        
        return {
            'buffer_size': len(self.level_buffer),
            'total_levels_created': self.level_counter,
            'avg_regret': np.mean(list(self.regret_scores.values())),
            'max_regret': max(self.regret_scores.values()),
            'min_regret': min(self.regret_scores.values()),
        }


# Domain-specific mutation functions

def highway_mutation_fn(level_config, mutation_rate=0.1):
    """
    Mutation function for highway-env
    Randomly modifies environment parameters
    """
    new_config = copy.deepcopy(level_config)
    
    # Each parameter has mutation_rate chance of mutating
    if np.random.random() < mutation_rate:
        # Mutate lanes
        delta = np.random.choice([-1, 0, 1])
        new_config['lanes_count'] = np.clip(
            new_config['lanes_count'] + delta, 2, 5
        )
    
    if np.random.random() < mutation_rate:
        # Mutate vehicle count
        delta = np.random.randint(-5, 6)
        new_config['vehicles_count'] = np.clip(
            new_config['vehicles_count'] + delta, 10, 50
        )
    
    if np.random.random() < mutation_rate:
        # Mutate density
        delta = np.random.uniform(-0.3, 0.3)
        new_config['vehicles_density'] = np.clip(
            new_config['vehicles_density'] + delta, 0.5, 3.0
        )
    
    if np.random.random() < mutation_rate:
        # Mutate duration
        delta = np.random.randint(-10, 11)
        new_config['duration'] = np.clip(
            new_config['duration'] + delta, 30, 100
        )
    
    return new_config


# Complete ACCEL training loop
def train_with_accel():
    """Train agent using ACCEL curriculum"""
    
    # Simple initial level
    initial_level = {
        'lanes_count': 3,
        'vehicles_count': 15,
        'vehicles_density': 0.8,
        'duration': 60,
    }
    
    # Initialize ACCEL
    accel = ACCELCurriculum(
        initial_level=initial_level,
        mutation_fn=highway_mutation_fn,
        buffer_size=100,
        regret_estimator='advantage',
        temperature=0.1,
    )
    
    # Create initial environment
    env = gymnasium.make("highway-v0", config=initial_level, render_mode='rgb_array')
    model = DQN('MlpPolicy', env, verbose=1)
    
    # Training loop
    total_timesteps = 200000
    timesteps_per_level = 2048
    
    timesteps = 0
    episode_data = []
    
    while timesteps < total_timesteps:
        # Sample and mutate level
        level_id, level_config = accel.sample_and_mutate()
        
        # Create environment with new level
        env = gymnasium.make("highway-v0", config=level_config, render_mode='rgb_array')
        model.set_env(env)
        
        # Collect episode data for regret calculation
        obs, _ = env.reset()
        episode_rewards = []
        episode_values = []
        episode_advantages = []
        done = truncated = False
        
        step_count = 0
        while not (done or truncated) and step_count < timesteps_per_level:
            # Predict action
            action, _ = model.predict(obs, deterministic=False)
            
            # Get value estimate
            with torch.no_grad():
                value = model.policy.predict_values(
                    torch.FloatTensor(obs).unsqueeze(0)
                )[0].item()
            
            # Step environment
            next_obs, reward, done, truncated, info = env.step(action)
            
            # Get next value
            with torch.no_grad():
                next_value = model.policy.predict_values(
                    torch.FloatTensor(next_obs).unsqueeze(0)
                )[0].item()
            
            # Compute advantage (TD error)
            advantage = reward + model.gamma * next_value * (1 - done) - value
            
            episode_rewards.append(reward)
            episode_values.append(value)
            episode_advantages.append(advantage)
            
            obs = next_obs
            step_count += 1
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(episode_rewards):
            R = r + model.gamma * R
            returns.insert(0, R)
        
        # Update regret for this level
        accel.update_regret(
            level_id,
            np.array(episode_advantages),
            np.array(episode_values),
            np.array(returns)
        )
        
        # Train model on collected data
        model.learn(step_count, reset_num_timesteps=False)
        
        timesteps += step_count
        
        # Log stats
        if timesteps % 10000 == 0:
            print(f"Timesteps: {timesteps}")
            print(f"ACCEL stats: {accel.get_stats()}")
            print(f"Current level: {level_config}")
            print("-" * 50)
    
    return model, accel
```

---

## 4. Comparison and Recommendations

### When to Use Each Method

**Domain Randomization**:
- ✅ Quick baseline for generalization
- ✅ Unknown target environment distribution
- ✅ Simple to implement
- ❌ Sample inefficient
- ❌ No automatic curriculum

**PLR**:
- ✅ Fixed set of known environment variations
- ✅ Want automatic curriculum
- ✅ Need sample efficiency
- ✅ Compatible with any RL algorithm
- ❌ Can't generate new environments
- ❌ Requires level tracking overhead

**ACCEL**:
- ✅ Need best possible generalization
- ✅ Can define level mutation operators
- ✅ Open-ended learning desired
- ✅ Have computational resources
- ❌ Most complex to implement
- ❌ Domain-specific mutations needed
- ❌ Computationally expensive

### Progression Path

1. **Start**: Domain Randomization
   - Get baseline generalization
   - Understand your environment space
   - Quick results

2. **Next**: PLR
   - Improve sample efficiency
   - Automatic curriculum
   - Define good evaluation levels

3. **Advanced**: ACCEL
   - Maximum generalization
   - Design mutation operators
   - Open-ended complexity

### For Your Highway Environment

**Recommendation**: Start with **PLR**

Why:
1. Highway variations are discrete and enumerable
2. You want generalization across traffic/lane configs
3. PLR will automatically find challenging configurations
4. Can later extend to ACCEL with mutations

**Specific Approach**:
```python
# 1. Define comprehensive environment space
lane_counts = [2, 3, 4]
vehicle_densities = [0.5, 1.0, 1.5, 2.0, 2.5]
vehicle_counts = range(10, 50, 5)
# = 3 * 5 * 8 = 120 configurations

# 2. Use PLR to find most useful configurations
# 3. Evaluate on held-out test configurations
# 4. Later add ACCEL mutations for continuous variation
```

---

## 5. Practical Tips

### Evaluation Strategy

```python
# Always split into train/test levels
train_configs = env_configs[:int(0.8 * len(env_configs))]
test_configs = env_configs[int(0.8 * len(env_configs)):]

# Periodic evaluation on test set
def evaluate_generalization(model, test_configs, n_episodes=10):
    """Evaluate zero-shot generalization"""
    results = []
    for config in test_configs:
        env = gym.make("highway-v0", config=config)
        episode_rewards = []
        
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done = truncated = False
            total_reward = 0
            
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += reward
            
            episode_rewards.append(total_reward)
        
        results.append(np.mean(episode_rewards))
    
    return {
        'mean_return': np.mean(results),
        'std_return': np.std(results),
        'min_return': np.min(results),
        'max_return': np.max(results),
    }
```

### Hyperparameter Guidelines

**Domain Randomization**:
- Start with narrow ranges, expand gradually
- Monitor training stability
- Use ADR if possible

**PLR**:
- `replay_prob`: 0.7-0.9 (higher = more curriculum)
- `temperature`: 0.05-0.2 (lower = more greedy)
- `staleness_coef`: 0.05-0.2
- `buffer_size`: 50-200 levels
- `score_function`: 'value_loss' usually works best

**ACCEL**:
- `mutation_rate`: 0.05-0.2 per parameter
- `temperature`: 0.05-0.15
- `buffer_size`: 100-500 levels
- Start with simple mutations, increase complexity

### Logging and Visualization

```python
# Track curriculum progression
import wandb

wandb.init(project="curriculum-learning")

# Log level difficulty over time
wandb.log({
    'curriculum/avg_lanes': config['lanes_count'],
    'curriculum/avg_density': config['vehicles_density'],
    'curriculum/regret_score': regret,
    'train/episode_reward': episode_reward,
})

# Visualize curriculum evolution
import matplotlib.pyplot as plt

def plot_curriculum_evolution(accel):
    """Plot how levels evolve over time"""
    level_ids = sorted(accel.levels.keys())
    
    lanes = [accel.levels[l]['lanes_count'] for l in level_ids]
    densities = [accel.levels[l]['vehicles_density'] for l in level_ids]
    
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(lanes)
    plt.title('Lane Count Evolution')
    plt.xlabel('Level ID')
    
    plt.subplot(1, 2, 2)
    plt.plot(densities)
    plt.title('Density Evolution')
    plt.xlabel('Level ID')
    
    plt.tight_layout()
    plt.savefig('curriculum_evolution.png')
```

---

## 6. Additional Resources

### Textbooks and Surveys

1. **"Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey"**
   - Authors: Narvekar et al.
   - Journal: JMLR 2020
   - URL: https://jmlr.org/papers/volume21/20-212/20-212.pdf
   - Comprehensive survey of curriculum learning

### Code Repositories

1. **PLR Official Implementation**:
   - https://github.com/facebookresearch/level-replay
   - PyTorch + PPO + Procgen

2. **ACCEL Implementation**:
   - https://github.com/minqi/pytorch-accel (check if exists)
   - Look for "evolving curricula" repositories

3. **Unsupervised Environment Design**:
   - https://github.com/facebookresearch/dcd
   - Dual Curriculum Design (combines PLR + PAIRED)

### Related Techniques

1. **Teacher-Student Methods**:
   - PAIRED, REPAIRED, Dual Curriculum Design
   - For adversarial curriculum generation

2. **Goal-Conditioned RL**:
   - Automatic goal curriculum
   - HER, CURIOUS, others

3. **Meta-Learning**:
   - MAML, Reptile for quick adaptation
   - Can complement curriculum approaches

---

## Summary

Your progression plan is excellent:

1. ✅ **Domain Randomization**: Simple baseline (1-2 days to implement)
2. ✅ **PLR**: Automatic curriculum (3-5 days to implement properly)
3. ✅ **ACCEL**: State-of-the-art generalization (1-2 weeks for full implementation)

This progression will give you:
- Strong baseline (DR)
- Sample-efficient curriculum (PLR)
- Open-ended complexity growth (ACCEL)

Start with PLR since it's most applicable to your discrete highway environment, then extend to ACCEL once you understand the learning dynamics.
