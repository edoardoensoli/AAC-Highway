"""
Prioritized Level Replay (PLR) for DQN + Highway-Env
=====================================================

Adapted from: https://github.com/facebookresearch/level-replay
Original paper: Jiang et al. (2021) "Prioritized Level Replay"

This implementation adapts the official PLR algorithm to work with:
- DQN (instead of PPO)
- Highway-env (instead of Procgen)
- Stable Baselines3
"""

import numpy as np
import torch
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import pickle


class LevelSampler:
    """
    Core PLR level sampling logic based on the official implementation.
    
    Key features from official repo:
    - value_l1: L1 value loss scoring
    - rank transform: Rank-based prioritization
    - Staleness correction
    - Replay vs new level sampling
    """
    
    def __init__(
        self,
        seeds: List[int],
        obs_space,
        action_space,
        num_actors: int = 1,
        strategy: str = 'value_l1',
        replay_schedule: str = 'fixed',
        score_transform: str = 'rank',
        temperature: float = 0.1,
        eps: float = 0.05,
        rho: float = 0.8,  # Replay probability
        nu: float = 0.5,   # Probability of sampling new unseen level
        alpha: float = 1.0,
        staleness_coef: float = 0.1,
        staleness_transform: str = 'power',
        staleness_temperature: float = 1.0,
    ):
        """
        Args:
            seeds: List of environment configuration indices/seeds
            obs_space: Observation space
            action_space: Action space
            num_actors: Number of parallel environments
            strategy: Scoring strategy ('value_l1', 'gae', 'one_step_td')
            replay_schedule: How replay probability changes ('fixed', 'proportional')
            score_transform: Transform for scores ('rank', 'power', 'softmax')
            temperature: Softmax temperature for sampling
            eps: Minimum sampling probability
            rho: Fixed replay probability
            nu: Probability of sampling new level when not replaying
            alpha: Power for power transform
            staleness_coef: Coefficient for staleness bonus
            staleness_transform: Transform for staleness ('power', 'constant')
            staleness_temperature: Temperature for staleness softmax
        """
        self.seeds = seeds
        self.obs_space = obs_space
        self.action_space = action_space
        self.num_actors = num_actors
        self.strategy = strategy
        self.replay_schedule = replay_schedule
        self.score_transform = score_transform
        self.temperature = temperature
        self.eps = eps
        self.rho = rho
        self.nu = nu
        self.alpha = alpha
        self.staleness_coef = staleness_coef
        self.staleness_transform = staleness_transform
        self.staleness_temperature = staleness_temperature
        
        # Level tracking
        self.unseen_seed_indices = set(range(len(seeds)))
        self.seed_scores = defaultdict(float)
        self.seed_staleness = defaultdict(int)
        self.partial_seed_scores = defaultdict(float)
        self.partial_seed_steps = defaultdict(int)
        
        # Statistics
        self.next_seed_index = 0
        self.seed_buffer = []  # Top-K seeds by score
        
        print(f"PLR LevelSampler initialized:")
        print(f"  Strategy: {strategy}")
        print(f"  Score transform: {score_transform}")
        print(f"  Temperature: {temperature}")
        print(f"  Staleness coef: {staleness_coef}")
        print(f"  Replay probability (rho): {rho}")
        print(f"  Total seeds: {len(seeds)}")
    
    def sample(self, strategy=None) -> Tuple[int, int]:
        """
        Sample next level/seed index.
        
        Returns:
            (seed_idx, seed): Tuple of index and actual seed/config
        """
        strategy = strategy or self.strategy
        
        # Determine if we replay or sample new
        if len(self.unseen_seed_indices) == 0:
            # All seen, always replay
            replay_decision = 1
        elif len(self.seed_scores) == 0:
            # No scores yet, sample new
            replay_decision = 0
        else:
            # Sample replay decision
            if self.replay_schedule == 'fixed':
                replay_decision = np.random.binomial(1, self.rho)
            elif self.replay_schedule == 'proportional':
                # Anneal as more levels seen
                seen_ratio = 1.0 - len(self.unseen_seed_indices) / len(self.seeds)
                replay_prob = self.rho * seen_ratio
                replay_decision = np.random.binomial(1, replay_prob)
            else:
                replay_decision = 1
        
        if replay_decision:
            # Sample from seen levels using scores
            seed_idx = self._sample_replay_level()
        else:
            # Sample new unseen level
            seed_idx = self._sample_unseen_level()
        
        # Update staleness
        for idx in self.seed_scores.keys():
            self.seed_staleness[idx] += 1
        self.seed_staleness[seed_idx] = 0
        
        seed = self.seeds[seed_idx]
        return seed_idx, seed
    
    def _sample_replay_level(self) -> int:
        """Sample from replay distribution using scores + staleness"""
        if len(self.seed_scores) == 0:
            # Fallback to unseen
            return self._sample_unseen_level()
        
        # Get scored seeds
        seed_indices = list(self.seed_scores.keys())
        scores = np.array([self.seed_scores[idx] for idx in seed_indices])
        
        # Apply score transform
        if self.score_transform == 'rank':
            # Rank-based (official PLR default)
            temp = np.flip(scores.argsort())
            ranks = np.empty_like(temp)
            ranks[temp] = np.arange(len(temp)) + 1
            weights = 1 / ranks
        elif self.score_transform == 'power':
            weights = np.power(scores, self.alpha)
        elif self.score_transform == 'softmax':
            weights = np.exp(scores / self.temperature)
        else:
            weights = scores
        
        # Add staleness bonus
        if self.staleness_coef > 0:
            staleness = np.array([self.seed_staleness[idx] for idx in seed_indices])
            
            if self.staleness_transform == 'power':
                staleness_weights = np.power(staleness, self.alpha)
            else:
                staleness_weights = staleness
            
            staleness_weights = np.exp(
                self.staleness_coef * staleness_weights / self.staleness_temperature
            )
            weights = weights * staleness_weights
        
        # Normalize to probabilities
        weights = weights / weights.sum()
        
        # Ensure minimum probability (from official repo)
        weights = (1 - self.eps) * weights + self.eps / len(weights)
        weights = weights / weights.sum()
        
        # Sample
        seed_idx = np.random.choice(seed_indices, p=weights)
        return seed_idx
    
    def _sample_unseen_level(self) -> int:
        """Sample uniformly from unseen levels"""
        if len(self.unseen_seed_indices) == 0:
            # All seen, sample from all
            return np.random.choice(len(self.seeds))
        
        seed_idx = np.random.choice(list(self.unseen_seed_indices))
        self.unseen_seed_indices.remove(seed_idx)
        return seed_idx
    
    def update_with_rollouts(self, rollouts: Dict[int, Dict]):
        """
        Update level scores with rollout data.
        
        Args:
            rollouts: Dict mapping seed_idx -> {
                'returns': np.array,
                'values': np.array,
                'advantages': np.array (optional for GAE),
            }
        """
        for seed_idx, data in rollouts.items():
            score = self._compute_score(data)
            
            # Update score (exponential moving average from official repo)
            if seed_idx in self.seed_scores:
                self.seed_scores[seed_idx] = 0.1 * score + 0.9 * self.seed_scores[seed_idx]
            else:
                self.seed_scores[seed_idx] = score
    
    def _compute_score(self, rollout_data: Dict) -> float:
        """
        Compute learning potential score for a level.
        
        Based on official repo strategies:
        - value_l1: L1 value loss (default and most common)
        - gae: Mean absolute GAE advantage
        - one_step_td: One-step TD error
        """
        returns = rollout_data['returns']
        values = rollout_data['values']
        
        if self.strategy == 'value_l1':
            # L1 value loss (most common in official repo)
            score = np.abs(returns - values).mean()
        
        elif self.strategy == 'gae':
            # GAE advantages if available
            if 'advantages' in rollout_data:
                advantages = rollout_data['advantages']
                score = np.abs(advantages).mean()
            else:
                # Fallback to value loss
                score = np.abs(returns - values).mean()
        
        elif self.strategy == 'one_step_td':
            # One-step TD error
            if 'td_errors' in rollout_data:
                score = np.abs(rollout_data['td_errors']).mean()
            else:
                # Approximate with value loss
                score = np.abs(returns - values).mean()
        
        else:
            # Default to value_l1
            score = np.abs(returns - values).mean()
        
        return float(score)
    
    def get_stats(self) -> Dict:
        """Get statistics for logging"""
        if len(self.seed_scores) == 0:
            return {
                'plr/num_seen_seeds': 0,
                'plr/num_unseen_seeds': len(self.unseen_seed_indices),
                'plr/mean_score': 0,
                'plr/max_score': 0,
            }
        
        scores = list(self.seed_scores.values())
        return {
            'plr/num_seen_seeds': len(self.seed_scores),
            'plr/num_unseen_seeds': len(self.unseen_seed_indices),
            'plr/mean_score': np.mean(scores),
            'plr/max_score': np.max(scores),
            'plr/min_score': np.min(scores),
            'plr/std_score': np.std(scores),
        }
    
    def save(self, filepath: str):
        """Save PLR state"""
        state = {
            'seed_scores': dict(self.seed_scores),
            'seed_staleness': dict(self.seed_staleness),
            'unseen_seed_indices': self.unseen_seed_indices,
            'next_seed_index': self.next_seed_index,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: str):
        """Load PLR state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.seed_scores = defaultdict(float, state['seed_scores'])
        self.seed_staleness = defaultdict(int, state['seed_staleness'])
        self.unseen_seed_indices = state['unseen_seed_indices']
        self.next_seed_index = state['next_seed_index']


class PLRWrapper:
    """
    Wrapper for integrating PLR with Stable Baselines3 DQN.
    
    This provides a simple interface similar to the official repo.
    """
    
    def __init__(
        self,
        env_configs: List[Dict],
        strategy: str = 'value_l1',
        score_transform: str = 'rank',
        temperature: float = 0.1,
        rho: float = 0.8,
        staleness_coef: float = 0.1,
    ):
        """
        Args:
            env_configs: List of environment configurations
            strategy: Scoring strategy
            score_transform: Transform for scores
            temperature: Softmax temperature
            rho: Replay probability
            staleness_coef: Staleness coefficient
        """
        self.env_configs = env_configs
        
        # Create dummy spaces (highway-env specific)
        # These are placeholders - actual spaces from env
        self.obs_space = None
        self.action_space = None
        
        # Initialize level sampler
        self.level_sampler = LevelSampler(
            seeds=list(range(len(env_configs))),
            obs_space=self.obs_space,
            action_space=self.action_space,
            strategy=strategy,
            score_transform=score_transform,
            temperature=temperature,
            rho=rho,
            staleness_coef=staleness_coef,
        )
        
        print(f"\n{'='*60}")
        print("PLR INITIALIZED")
        print(f"{'='*60}")
        print(f"Total configurations: {len(env_configs)}")
        print(f"Strategy: {strategy}")
        print(f"Replay probability: {rho}")
        print(f"{'='*60}\n")
    
    def sample_config(self) -> Tuple[int, Dict]:
        """Sample next configuration"""
        seed_idx, _ = self.level_sampler.sample()
        config = self.env_configs[seed_idx]
        return seed_idx, config
    
    def update_with_episode(
        self,
        seed_idx: int,
        returns: np.ndarray,
        values: np.ndarray,
        advantages: Optional[np.ndarray] = None
    ):
        """
        Update PLR scores after episode.
        
        Args:
            seed_idx: Configuration index
            returns: Episode returns
            values: Value predictions
            advantages: GAE advantages (optional)
        """
        rollout_data = {
            'returns': returns,
            'values': values,
        }
        
        if advantages is not None:
            rollout_data['advantages'] = advantages
        
        rollouts = {seed_idx: rollout_data}
        self.level_sampler.update_with_rollouts(rollouts)
    
    def get_stats(self) -> Dict:
        """Get PLR statistics"""
        return self.level_sampler.get_stats()
    
    def save(self, filepath: str):
        """Save PLR state"""
        self.level_sampler.save(filepath)
    
    def load(self, filepath: str):
        """Load PLR state"""
        self.level_sampler.load(filepath)


# Utility function for collecting episode data for PLR
def collect_episode_for_plr(env, model, max_steps=2000):
    """
    Collect episode data needed for PLR scoring.
    
    Returns:
        Dict with 'returns', 'values', 'advantages'
    """
    obs, _ = env.reset()
    
    states = []
    actions = []
    rewards = []
    values = []
    dones = []
    
    done = truncated = False
    steps = 0
    
    while not (done or truncated) and steps < max_steps:
        # Get value prediction
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
            value = model.policy.predict_values(obs_tensor)[0].item()
        
        # Get action
        action, _ = model.predict(obs, deterministic=False)
        
        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)
        
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        values.append(value)
        dones.append(done or truncated)
        
        obs = next_obs
        steps += 1
    
    # Compute returns (Monte Carlo)
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + model.gamma * R
        returns.insert(0, R)
    
    returns = np.array(returns)
    values = np.array(values)
    
    # Compute advantages (TD residual approximation)
    advantages = returns - values
    
    return {
        'returns': returns,
        'values': values,
        'advantages': advantages,
        'episode_reward': sum(rewards),
        'episode_length': len(rewards),
    }


if __name__ == "__main__":
    # Example usage
    print("PLR Implementation Test")
    print("="*60)
    
    # Create dummy configs
    configs = []
    for lanes in [2, 3, 4]:
        for density in [0.8, 1.0, 1.5, 2.0]:
            for vehicles in [15, 20, 25, 30]:
                config = {
                    'lanes_count': lanes,
                    'vehicles_density': density,
                    'vehicles_count': vehicles,
                }
                configs.append(config)
    
    print(f"Created {len(configs)} configurations")
    
    # Initialize PLR
    plr = PLRWrapper(
        env_configs=configs,
        strategy='value_l1',
        score_transform='rank',
        rho=0.8,
    )
    
    # Simulate sampling
    print("\nSampling 10 configurations:")
    for i in range(10):
        seed_idx, config = plr.sample_config()
        print(f"  {i+1}. Config {seed_idx}: {config}")
        
        # Simulate episode data
        dummy_returns = np.random.randn(100) * 10
        dummy_values = np.random.randn(100) * 10
        
        plr.update_with_episode(seed_idx, dummy_returns, dummy_values)
    
    # Print stats
    print("\nPLR Statistics:")
    stats = plr.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n" + "="*60)
    print("Test complete!")
