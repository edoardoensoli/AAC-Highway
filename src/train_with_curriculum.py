"""
Training script with curriculum learning for Highway-Env
Supports: Domain Randomization (DR) and Prioritized Level Replay (PLR)
"""

import gymnasium
import highway_env
import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
import argparse
import os
import json
from datetime import datetime

from plr_implementation import PLRManager, collect_episode_data


class DomainRandomization:
    """Simple Domain Randomization baseline"""
    
    def __init__(self, env_id="highway-v0"):
        self.env_id = env_id
    
    def sample_config(self):
        """Sample random configuration"""
        return {
            'lanes_count': np.random.choice([2, 3, 4]),
            'vehicles_count': np.random.randint(15, 40),
            'vehicles_density': np.random.uniform(0.8, 2.5),
            'duration': 60,
            'simulation_frequency': 30,
        }
    
    def create_env(self):
        """Create environment with random config"""
        config = self.sample_config()
        return gymnasium.make(self.env_id, config=config, render_mode='rgb_array'), config


class CurriculumCallback(BaseCallback):
    """Callback for curriculum learning during training"""
    
    def __init__(
        self,
        curriculum_manager,
        curriculum_type='plr',
        update_interval=2048,
        eval_freq=10000,
        test_configs=None,
        log_dir='logs',
    ):
        super().__init__()
        self.curriculum_manager = curriculum_manager
        self.curriculum_type = curriculum_type
        self.update_interval = update_interval
        self.eval_freq = eval_freq
        self.test_configs = test_configs or []
        self.log_dir = log_dir
        
        self.current_level_id = None
        self.steps_in_level = 0
        self.episode_rewards = []
        self.episode_values = []
        
        os.makedirs(log_dir, exist_ok=True)
    
    def _on_training_start(self):
        """Called before training starts"""
        if self.curriculum_type == 'plr':
            # Sample initial level
            self.current_level_id, config = self.curriculum_manager.sample_level()
            env = gymnasium.make("highway-v0", config=config, render_mode='rgb_array')
            self.model.set_env(env)
    
    def _on_step(self) -> bool:
        """Called at each step"""
        self.steps_in_level += 1
        
        # Update environment periodically based on curriculum
        if self.steps_in_level >= self.update_interval:
            self._update_curriculum()
            self.steps_in_level = 0
        
        # Periodic evaluation
        if self.num_timesteps % self.eval_freq == 0:
            self._evaluate()
        
        return True
    
    def _update_curriculum(self):
        """Update curriculum and switch environment"""
        if self.curriculum_type == 'dr':
            # Domain Randomization: just sample new random config
            env, config = self.curriculum_manager.create_env()
            self.model.set_env(env)
            
        elif self.curriculum_type == 'plr':
            # PLR: collect data and update scores
            episode_data = collect_episode_data(
                self.training_env,
                self.model,
                max_steps=min(1000, self.update_interval)
            )
            
            # Update PLR score for current level
            if self.current_level_id is not None:
                self.curriculum_manager.update_level_score(
                    self.current_level_id,
                    episode_data
                )
            
            # Sample next level
            self.current_level_id, config = self.curriculum_manager.sample_level()
            env = gymnasium.make("highway-v0", config=config, render_mode='rgb_array')
            self.model.set_env(env)
            
            # Log PLR stats
            stats = self.curriculum_manager.get_stats()
            for key, value in stats.items():
                self.logger.record(key, value)
    
    def _evaluate(self):
        """Evaluate on held-out test configurations"""
        if len(self.test_configs) == 0:
            return
        
        test_rewards = []
        for config in self.test_configs:
            env = gymnasium.make("highway-v0", config=config, render_mode='rgb_array')
            
            # Run episode
            obs, _ = env.reset()
            done = truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
            
            test_rewards.append(episode_reward)
            env.close()
        
        # Log evaluation results
        self.logger.record('eval/mean_reward', np.mean(test_rewards))
        self.logger.record('eval/std_reward', np.std(test_rewards))
        self.logger.record('eval/min_reward', np.min(test_rewards))
        self.logger.record('eval/max_reward', np.max(test_rewards))
        
        print(f"\nEvaluation at step {self.num_timesteps}:")
        print(f"  Mean reward: {np.mean(test_rewards):.2f} +/- {np.std(test_rewards):.2f}")


def create_train_test_split(num_train=80, num_test=20):
    """Create train/test environment configurations"""
    all_configs = []
    
    for lanes in [2, 3, 4]:
        for density in np.linspace(0.8, 2.2, 5):
            for vehicles in range(15, 40, 5):
                config = {
                    'lanes_count': lanes,
                    'vehicles_density': float(density),
                    'vehicles_count': int(vehicles),
                    'duration': 60,
                    'simulation_frequency': 30,
                }
                all_configs.append(config)
    
    # Shuffle and split
    np.random.shuffle(all_configs)
    train_configs = all_configs[:num_train]
    test_configs = all_configs[num_train:num_train + num_test]
    
    return train_configs, test_configs


def train_baseline(args):
    """Train baseline DQN without curriculum"""
    print("Training baseline DQN (single environment)...")
    
    # Fixed environment
    config = {
        'lanes_count': 3,
        'vehicles_count': 25,
        'vehicles_density': 1.0,
        'duration': 60,
        'simulation_frequency': 30,
    }
    
    env = gymnasium.make("highway-v0", config=config, render_mode='rgb_array')
    
    # Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    model = DQN(
        'MlpPolicy',
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log=f"{args.log_dir}/baseline/",
        device=device
    )
    
    # Train
    model.learn(total_timesteps=args.total_timesteps)
    
    # Save
    os.makedirs(f"{args.log_dir}/baseline", exist_ok=True)
    model.save(f"{args.log_dir}/baseline/model")
    
    return model


def train_domain_randomization(args):
    """Train with Domain Randomization"""
    print("Training with Domain Randomization...")
    
    # Create DR manager
    dr = DomainRandomization()
    env, config = dr.create_env()
    
    # Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create model
    model = DQN(
        'MlpPolicy',
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log=f"{args.log_dir}/dr/",
        device=device
    )
    
    # Create test configs
    _, test_configs = create_train_test_split()
    
    # Callback for environment switching
    callback = CurriculumCallback(
        curriculum_manager=dr,
        curriculum_type='dr',
        update_interval=2048,
        test_configs=test_configs,
        log_dir=f"{args.log_dir}/dr"
    )
    
    # Train
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    
    # Save
    os.makedirs(f"{args.log_dir}/dr", exist_ok=True)
    model.save(f"{args.log_dir}/dr/model")
    
    return model


def train_plr(args):
    """Train with Prioritized Level Replay"""
    print("Training with PLR...")
    
    # Create train/test split
    train_configs, test_configs = create_train_test_split(num_train=80, num_test=20)
    
    # Initialize PLR
    plr = PLRManager(
        env_id="highway-v0",
        train_env_configs=train_configs,
        score_function='value_loss',
        replay_probability=0.8,
        temperature=0.1,
        staleness_coef=0.1,
        buffer_size=50,
    )
    
    # Sample initial level
    level_id, config = plr.sample_level()
    env = gymnasium.make("highway-v0", config=config, render_mode='rgb_array')
    
    # Device detection
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Training on {len(train_configs)} configurations")
    print(f"Testing on {len(test_configs)} held-out configurations")
    
    # Create model
    model = DQN(
        'MlpPolicy',
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log=f"{args.log_dir}/plr/",
        device=device
    )
    
    # Callback for PLR
    callback = CurriculumCallback(
        curriculum_manager=plr,
        curriculum_type='plr',
        update_interval=2048,
        test_configs=test_configs,
        log_dir=f"{args.log_dir}/plr"
    )
    
    # Train
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    
    # Save
    os.makedirs(f"{args.log_dir}/plr", exist_ok=True)
    model.save(f"{args.log_dir}/plr/model")
    plr.save(f"{args.log_dir}/plr/plr_state.pkl")
    
    # Save configs
    with open(f"{args.log_dir}/plr/train_configs.json", 'w') as f:
        json.dump(train_configs, f, indent=2)
    with open(f"{args.log_dir}/plr/test_configs.json", 'w') as f:
        json.dump(test_configs, f, indent=2)
    
    return model, plr


def main():
    parser = argparse.ArgumentParser(description='Train Highway-Env with Curriculum Learning')
    parser.add_argument('--method', type=str, default='plr',
                        choices=['baseline', 'dr', 'plr'],
                        help='Training method: baseline, dr (domain randomization), or plr')
    parser.add_argument('--total-timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--log-dir', type=str, default='highway_curriculum_logs',
                        help='Directory for logs and saved models')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create log directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.log_dir = f"{args.log_dir}_{timestamp}"
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Save args
    with open(f"{args.log_dir}/args.json", 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Train based on method
    if args.method == 'baseline':
        model = train_baseline(args)
    elif args.method == 'dr':
        model = train_domain_randomization(args)
    elif args.method == 'plr':
        model, plr = train_plr(args)
    
    print(f"\nTraining complete! Model saved to {args.log_dir}/{args.method}/")


if __name__ == "__main__":
    main()
