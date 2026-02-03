"""
Prioritized Level Replay (PLR) Implementation for Highway-Env
Based on Jiang et al. 2021 - https://arxiv.org/abs/2010.03934

This implementation is designed to work with Stable Baselines3 DQN.
"""

import numpy as np
import gymnasium
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import torch
import pickle
import os


class PLRManager:
    """
    Manages curriculum learning using Prioritized Level Replay.
    
    Features:
    - Maintains buffer of high-learning-potential environments
    - Scores environments based on TD-error or value loss
    - Implements staleness-aware sampling
    - Automatic curriculum from easy to hard
    """
    
    def __init__(
        self,
        env_id: str = "highway-v0",
        train_env_configs: Optional[List[Dict]] = None,
        score_function: str = 'value_loss',
        replay_probability: float = 0.8,
        temperature: float = 0.1,
        staleness_coef: float = 0.1,
        buffer_size: int = 100,
        score_transform: str = 'rank',
    ):
        """
        Args:
            env_id: Gymnasium environment ID
            train_env_configs: List of environment configurations to sample from
            score_function: How to score levels ('value_loss', 'advantage', 'return')
            replay_probability: Prob of sampling from buffer vs new level
            temperature: Temperature for sampling (lower = more greedy)
            staleness_coef: Weight for staleness bonus
            buffer_size: Maximum number of levels to track
            score_transform: How to transform scores before sampling
        """
        self.env_id = env_id
        self.score_function = score_function
        self.replay_prob = replay_probability
        self.temperature = temperature
        self.staleness_coef = staleness_coef
        self.buffer_size = buffer_size
        self.score_transform = score_transform
        
        # Generate or use provided configs
        if train_env_configs is None:
            self.env_configs = self._generate_default_configs()
        else:
            self.env_configs = train_env_configs
        
        # Tracking structures
        self.scores = defaultdict(lambda: 1.0)
        self.staleness = defaultdict(int)
        self.visit_counts = defaultdict(int)
        self.seen_level_ids = set()
        self.level_buffer = []
        
        self.global_step = 0
        self.total_levels = len(self.env_configs)
        
        print(f"PLR initialized with {self.total_levels} environment configurations")
    
    def _generate_default_configs(self) -> List[Dict]:
        """Generate diverse highway environment configurations"""
        configs = []
        
        for lanes in [2, 3, 4]:
            for density in [0.8, 1.0, 1.5, 2.0]:
                for vehicles in [15, 20, 25, 30, 35]:
                    config = {
                        'lanes_count': lanes,
                        'vehicles_density': density,
                        'vehicles_count': vehicles,
                        'duration': 60,
                        'simulation_frequency': 30,
                    }
                    configs.append(config)
        
        return configs
    
    def sample_level(self) -> Tuple[int, Dict]:
        """Sample next training level using PLR strategy"""
        self.global_step += 1
        
        if len(self.level_buffer) > 0 and np.random.random() < self.replay_prob:
            level_id = self._sample_from_buffer()
        else:
            level_id = self._sample_new_level()
        
        # Update staleness
        for lid in self.level_buffer:
            self.staleness[lid] += 1
        self.staleness[level_id] = 0
        
        self.visit_counts[level_id] += 1
        self.seen_level_ids.add(level_id)
        
        return level_id, self.env_configs[level_id]
    
    def _sample_from_buffer(self) -> int:
        """Sample level from buffer using prioritization"""
        weights = []
        for level_id in self.level_buffer:
            score = self.scores[level_id]
            staleness_bonus = self.staleness_coef * self.staleness[level_id]
            weights.append(score + staleness_bonus)
        
        weights = np.array(weights)
        
        if self.score_transform == 'rank':
            ranks = np.argsort(np.argsort(weights)) + 1
            weights = ranks
        elif self.score_transform == 'power':
            weights = np.power(weights, 2)
        
        if self.temperature > 0:
            probs = np.exp(weights / self.temperature)
            probs = probs / np.sum(probs)
        else:
            probs = np.zeros(len(weights))
            probs[np.argmax(weights)] = 1.0
        
        idx = np.random.choice(len(self.level_buffer), p=probs)
        return self.level_buffer[idx]
    
    def _sample_new_level(self) -> int:
        """Sample a new unseen level uniformly"""
        unseen = [i for i in range(self.total_levels) if i not in self.seen_level_ids]
        
        if len(unseen) > 0:
            return np.random.choice(unseen)
        else:
            return np.random.randint(0, self.total_levels)
    
    def update_level_score(
        self,
        level_id: int,
        episode_data: Dict[str, np.ndarray]
    ):
        """Update score for a level after collecting episode data"""
        
        advantages = episode_data.get('advantages', np.array([]))
        values = episode_data.get('values', np.array([]))
        returns = episode_data.get('returns', np.array([]))
        
        if len(advantages) == 0 or len(values) == 0:
            return
        
        # Compute score
        if self.score_function == 'value_loss':
            score = np.mean(np.abs(values - returns))
        elif self.score_function == 'advantage':
            score = np.mean(np.abs(advantages))
        elif self.score_function == 'positive_advantage':
            pos_adv = advantages[advantages > 0]
            score = np.mean(pos_adv) if len(pos_adv) > 0 else 0.0
        elif self.score_function == 'return':
            score = -np.mean(returns)  # Negative because low return = hard
        else:
            score = 1.0
        
        # Exponential moving average
        alpha = 0.1
        self.scores[level_id] = alpha * score + (1 - alpha) * self.scores[level_id]
        
        # Update buffer
        self._update_buffer(level_id)
    
    def _update_buffer(self, level_id: int):
        """Maintain top-K levels by score"""
        if level_id not in self.level_buffer:
            self.level_buffer.append(level_id)
        
        # Sort by score and trim
        self.level_buffer.sort(key=lambda l: self.scores[l], reverse=True)
        self.level_buffer = self.level_buffer[:self.buffer_size]
    
    def get_stats(self) -> Dict:
        """Get statistics for logging"""
        if len(self.scores) == 0:
            return {}
        
        scores_list = list(self.scores.values())
        return {
            'plr/num_seen_levels': len(self.seen_level_ids),
            'plr/buffer_size': len(self.level_buffer),
            'plr/avg_score': np.mean(scores_list),
            'plr/max_score': np.max(scores_list),
            'plr/min_score': np.min(scores_list),
            'plr/seen_fraction': len(self.seen_level_ids) / self.total_levels,
        }
    
    def save(self, filepath: str):
        """Save PLR state"""
        state = {
            'scores': dict(self.scores),
            'staleness': dict(self.staleness),
            'visit_counts': dict(self.visit_counts),
            'seen_level_ids': self.seen_level_ids,
            'level_buffer': self.level_buffer,
            'global_step': self.global_step,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: str):
        """Load PLR state"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.scores = defaultdict(lambda: 1.0, state['scores'])
        self.staleness = defaultdict(int, state['staleness'])
        self.visit_counts = defaultdict(int, state['visit_counts'])
        self.seen_level_ids = state['seen_level_ids']
        self.level_buffer = state['level_buffer']
        self.global_step = state['global_step']


def collect_episode_data(env, model, max_steps=1000):
    """
    Collect episode data for PLR scoring.
    
    Returns:
        episode_data: Dict with 'advantages', 'values', 'returns'
    """
    obs, _ = env.reset()
    
    values = []
    rewards = []
    dones = []
    
    done = truncated = False
    steps = 0
    
    while not (done or truncated) and steps < max_steps:
        # Get value prediction
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
            value = model.policy.predict_values(obs_tensor)[0].item()
        
        # Predict action and step
        action, _ = model.predict(obs, deterministic=False)
        next_obs, reward, done, truncated, info = env.step(action)
        
        values.append(value)
        rewards.append(reward)
        dones.append(done or truncated)
        
        obs = next_obs
        steps += 1
    
    # Compute returns
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + model.gamma * R
        returns.insert(0, R)
    
    returns = np.array(returns)
    values = np.array(values)
    
    # Compute advantages (simplified GAE)
    advantages = returns - values
    
    return {
        'advantages': advantages,
        'values': values,
        'returns': returns,
        'episode_reward': sum(rewards),
        'episode_length': len(rewards),
    }


# Example usage
if __name__ == "__main__":
    # This shows how to integrate PLR with your DQN training
    from stable_baselines3 import DQN
    
    # Initialize PLR
    plr = PLRManager(
        env_id="highway-v0",
        score_function='value_loss',
        replay_probability=0.8,
        temperature=0.1,
    )
    
    # Sample first level
    level_id, config = plr.sample_level()
    env = gymnasium.make("highway-v0", config=config, render_mode='rgb_array')
    
    # Create model
    model = DQN('MlpPolicy', env, verbose=1)
    
    # Training loop
    total_timesteps = 100000
    update_interval = 2048
    
    timesteps = 0
    while timesteps < total_timesteps:
        # Collect episode data
        episode_data = collect_episode_data(env, model, max_steps=1000)
        
        # Update PLR score
        plr.update_level_score(level_id, episode_data)
        
        # Train model
        model.learn(update_interval, reset_num_timesteps=False)
        timesteps += update_interval
        
        # Sample next level
        level_id, config = plr.sample_level()
        env = gymnasium.make("highway-v0", config=config, render_mode='rgb_array')
        model.set_env(env)
        
        # Log stats
        if timesteps % 10000 == 0:
            stats = plr.get_stats()
            print(f"\nTimestep {timesteps}:")
            print(f"  Episode reward: {episode_data['episode_reward']:.2f}")
            for key, value in stats.items():
                print(f"  {key}: {value:.4f}")
    
    # Save trained model and PLR state
    model.save("highway_dqn_plr/model")
    plr.save("highway_dqn_plr/plr_state.pkl")
