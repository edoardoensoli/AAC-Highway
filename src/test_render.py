#!/usr/bin/env python
"""
Test DQN models on highway-env with visual rendering.
Displays agent behavior across different difficulty levels and tracks proximity metrics.
"""

import gymnasium
import numpy as np
from pathlib import Path
import json
from stable_baselines3 import DQN
from src.dqn_accel import ACCELGenerator

def _compute_min_distance(env):
    """Compute minimum distance from ego vehicle to nearby vehicles (front/side)."""
    try:
        ego = env.unwrapped.vehicle
        road = env.unwrapped.road
        if ego is None or road is None:
            return float('inf')
    except Exception:
        return float('inf')

    min_dist = float('inf')
    for v in road.vehicles:
        if v is ego:
            continue
        dx = v.position[0] - ego.position[0]
        dy = v.position[1] - ego.position[1]
        if dx > -5.0 and abs(dy) < 8.0:
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                min_dist = dist
    return min_dist

def test_model_with_render(
    model_path: str,
    n_episodes: int = 5,
    device: str = 'cpu',
    difficulty: str = 'easy',
):
    """
    Test model with visual rendering.
    
    Args:
        model_path: Path to model .zip file
        n_episodes: Number of episodes to render
        device: cpu or cuda
        difficulty: easy, medium, hard, or expert
    """
    
    print(f"\n{'='*65}")
    print(f"  Testing with render: {Path(model_path).name}")
    print(f"{'='*65}\n")
    
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return
    
    model = DQN.load(model_path, device=device)
    print(f"Model loaded from: {model_path}")
    
    configs = {
        'easy': {
            'name': 'Easy: 2 lanes, 30 vehicles, IDM (Stage 0)',
            'lanes_count': 2, 'vehicles_count': 30, 'vehicles_density': 1.0, 'duration': 50,
            'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
        },
        'medium': {
            'name': 'Medium: 3 lanes, 20 vehicles, IDM (Stage 2)',
            'lanes_count': 3, 'vehicles_count': 20, 'vehicles_density': 1.0, 'duration': 100,
            'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
        },
        'hard': {
            'name': 'Hard: 4 lanes, 40 vehicles, Aggressive (Stage 3)',
            'lanes_count': 4, 'vehicles_count': 40, 'vehicles_density': 1.2, 'duration': 60,
            'other_vehicles_type': 'highway_env.vehicle.behavior.AggressiveVehicle',
        },
        'expert': {
            'name': 'Expert: 2 lanes, 50 vehicles, Aggressive (Stage 6)',
            'lanes_count': 2, 'vehicles_count': 50, 'vehicles_density': 2.0, 'duration': 80,
            'other_vehicles_type': 'highway_env.vehicle.behavior.AggressiveVehicle',
        }
    }
    
    if difficulty not in configs:
        print(f"Unknown difficulty: {difficulty}")
        print(f"Options: {', '.join(configs.keys())}")
        return
    
    cfg_info = configs[difficulty]
    config = {
        **ACCELGenerator.FIXED_PARAMS,
        **{k: v for k, v in cfg_info.items() if k != 'name'}
    }
    
    print(f"Difficulty: {cfg_info['name']}")
    print(f"Config: vehicles={config['vehicles_count']}, "
          f"density={config['vehicles_density']}, "
          f"lanes={config['lanes_count']}, "
          f"duration={config['duration']}")
    print(f"\nRunning {n_episodes} test episodes...\n")
    
    env = gymnasium.make('highway-v0', config=config, render_mode='human')
    
    episode_returns = []
    episode_lengths = []
    episode_min_distances = []
    episode_avg_distances = []
    
    try:
        for ep in range(n_episodes):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            ep_length = 0
            ep_distances = []
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_return += reward
                ep_length += 1
                
                min_d = _compute_min_distance(env)
                if min_d < 100.0:
                    ep_distances.append(min_d)
                
                env.render()
            
            episode_returns.append(ep_return)
            episode_lengths.append(ep_length)
            
            ep_min_dist = min(ep_distances) if ep_distances else float('inf')
            ep_avg_dist = np.mean(ep_distances) if ep_distances else float('inf')
            episode_min_distances.append(ep_min_dist)
            episode_avg_distances.append(ep_avg_dist)
            
            expected_steps = config['duration'] * config.get('policy_frequency', 2) * 0.85
            survived = "SURVIVED" if ep_length >= expected_steps else "CRASHED"
            
            dist_warning = ""
            if ep_min_dist < 10:
                dist_warning = " [TOO CLOSE]"
            elif ep_min_dist < 15:
                dist_warning = " [low distance]"
            
            print(f"Ep {ep+1:2d}/{n_episodes} | Return: {ep_return:7.2f} | "
                  f"Length: {ep_length:3d} | MinDist: {ep_min_dist:5.1f}m | "
                  f"AvgDist: {ep_avg_dist:5.1f}m | {survived}{dist_warning}")
    
    finally:
        env.close()
    
    print(f"\n{'='*65}")
    print(f"Test results: {cfg_info['name']}")
    print(f"{'='*65}")
    print(f"Average reward:      {np.mean(episode_returns):7.2f} ± {np.std(episode_returns):.2f}")
    print(f"Average length:      {np.mean(episode_lengths):7.0f} steps")
    survival_rate = sum(1 for l in episode_lengths if l >= config['duration'] * 2 * 0.85) / len(episode_lengths)
    print(f"Survival rate:       {survival_rate*100:6.1f}%")
    
    finite_mins = [d for d in episode_min_distances if d < float('inf')]
    finite_avgs = [d for d in episode_avg_distances if d < float('inf')]
    if finite_mins:
        print(f"Min distance:        {np.mean(finite_mins):5.1f}m (avg), {min(finite_mins):5.1f}m (worst)")
        print(f"Avg distance:        {np.mean(finite_avgs):5.1f}m")
    print(f"Episodes completed:  {len(episode_returns)}/{n_episodes}")
    print(f"{'='*65}\n")
    
    return {
        'difficulty': difficulty,
        'avg_reward': float(np.mean(episode_returns)),
        'avg_length': float(np.mean(episode_lengths)),
        'survival_rate': float(survival_rate),
        'episode_returns': [float(r) for r in episode_returns],
        'episode_lengths': [int(l) for l in episode_lengths],
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test DQN model with rendering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_render.py --model ./highway_dqn_accel/dqn_accel_final_1M.zip --difficulty easy
  python test_render.py --model ./highway_dqn_accel/dqn_accel_final_1M.zip --difficulty expert --episodes 10
  python test_render.py --model ./highway_dqn_accel/dqn_accel_final_1M.zip --all-difficulties
        """
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to DQN model (.zip)')
    parser.add_argument('--difficulty', type=str, default='easy',
                        choices=['easy', 'medium', 'hard', 'expert'],
                        help='Difficulty level (default: easy)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of episodes to render (default: 5)')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda'],
                        help='Device (default: cpu)')
    parser.add_argument('--all-difficulties', action='store_true',
                        help='Test all difficulty levels')
    
    args = parser.parse_args()
    
    if args.all_difficulties:
        results = {}
        for diff in ['easy', 'medium', 'hard', 'expert']:
            result = test_model_with_render(
                model_path=args.model,
                n_episodes=args.episodes,
                device=args.device,
                difficulty=diff
            )
            if result:
                results[diff] = result
        
        save_path = Path(args.model).parent / 'render_test_results.json'
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved: {save_path}")
    else:
        test_model_with_render(
            model_path=args.model,
            n_episodes=args.episodes,
            device=args.device,
            difficulty=args.difficulty
        )
