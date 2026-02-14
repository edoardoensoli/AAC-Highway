"""
Metrics system for evaluating RL model performance in highway-env.
Uses env.unwrapped to access real vehicle data.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass, field
import json
from datetime import datetime
from pathlib import Path


@dataclass
class EpisodeData:
    """Data collected during a single episode."""
    # Final state
    crashed: bool = False
    truncated: bool = False
    
    # Counters
    total_reward: float = 0.0
    steps: int = 0
    cars_overtaken: int = 0
    lane_changes: int = 0
    
    # Speed
    speeds: List[float] = field(default_factory=list)
    
    # Distance
    start_x: Optional[float] = None
    end_x: float = 0.0
    
    # Safety
    min_ttc: float = float('inf')  # Minimum Time To Collision
    near_miss_count: int = 0  # Near-miss situations
    
    # Internal tracking (not final metrics)
    _previous_lane: Optional[int] = None
    _overtaken_ids: Set[int] = field(default_factory=set)
    _vehicle_positions: Dict[int, float] = field(default_factory=dict)


class HighwayMetrics:
    """
    Metrics tracker for highway-env.
    
    Available metrics:
    - collision_rate: % of episodes ending in collision
    - survival_rate: % of episodes completed without crash
    - avg_reward: average reward per episode
    - avg_speed: average speed (m/s)
    - max_speed: maximum speed reached (m/s)
    - cars_overtaken: average cars overtaken per episode
    - total_cars_overtaken: total cars overtaken
    - avg_episode_length: average episode duration (steps)
    - lane_changes: average lane changes per episode
    - distance_traveled: average distance traveled (m)
    - min_ttc: average minimum Time To Collision (s) - safety metric
    - near_miss_rate: % of episodes with near-misses
    """
    
    AVAILABLE_METRICS = {
        'collision_rate',
        'survival_rate', 
        'avg_reward',
        'avg_speed',
        'max_speed',
        'cars_overtaken',
        'total_cars_overtaken',
        'avg_episode_length',
        'lane_changes',
        'distance_traveled',
        'min_ttc',
        'near_miss_rate',
    }
    
    def __init__(
        self,
        metrics: Optional[Set[str]] = None,
        overtake_threshold: float = 3.0,   # meters to consider an overtake
        near_miss_distance: float = 5.0,   # meters for near-miss
        ttc_threshold: float = 2.0,        # seconds for critical TTC
        verbose: bool = False
    ):
        """
        Args:
            metrics: Set of metrics to compute. None = all.
            overtake_threshold: Distance (m) to detect an overtake
            near_miss_distance: Distance (m) to consider a near-miss
            ttc_threshold: TTC threshold (s) for critical situations
            verbose: Print debug during evaluation
        """
        if metrics is None:
            self.active_metrics = self.AVAILABLE_METRICS.copy()
        else:
            invalid = metrics - self.AVAILABLE_METRICS
            if invalid:
                raise ValueError(f"Invalid metrics: {invalid}. Available: {self.AVAILABLE_METRICS}")
            self.active_metrics = metrics
        
        self.overtake_threshold = overtake_threshold
        self.near_miss_distance = near_miss_distance
        self.ttc_threshold = ttc_threshold
        self.verbose = verbose
        
        self.episodes: List[EpisodeData] = []
        self.current: Optional[EpisodeData] = None
    
    def start_episode(self, env: Any):
        """Start a new episode."""
        self.current = EpisodeData()
        
        # Save initial position
        try:
            ego = env.unwrapped.vehicle
            self.current.start_x = float(ego.position[0])
        except:
            self.current.start_x = 0.0
    
    def step(self, env: Any, action: int, reward: float, done: bool, truncated: bool, info: Dict):
        """
        Update metrics after each step.
        
        Args:
            env: The gymnasium environment (uses env.unwrapped)
            action: Action taken
            reward: Reward received
            done: Episode terminated
            truncated: Episode truncated
            info: Info from environment
        """
        if self.current is None:
            return
        
        ep = self.current
        ep.steps += 1
        ep.total_reward += reward
        
        # Read state from info (always available)
        ep.crashed = info.get('crashed', False)
        ep.truncated = truncated
        
        # Access real data via env.unwrapped
        try:
            u = env.unwrapped
            ego = u.vehicle
            ego_x = float(ego.position[0])
            ego_y = float(ego.position[1])
            ego_speed = float(ego.speed)
            ego_vx = float(ego.velocity[0]) if hasattr(ego, 'velocity') else ego_speed
            
            # Update final position
            ep.end_x = ego_x
            
            # Speed
            ep.speeds.append(ego_speed)
            
            # Lane changes
            lane_idx = ego.lane_index
            if isinstance(lane_idx, tuple):
                current_lane = int(lane_idx[-1])
            else:
                current_lane = 0
            
            if ep._previous_lane is not None and current_lane != ep._previous_lane:
                ep.lane_changes += 1
            ep._previous_lane = current_lane
            
            # Analyze other vehicles
            for v in u.road.vehicles:
                if v is ego:
                    continue
                
                vid = id(v)
                v_x = float(v.position[0])
                v_y = float(v.position[1])
                v_vx = float(v.velocity[0]) if hasattr(v, 'velocity') else float(v.speed)
                
                # Relative distance
                rel_x = v_x - ego_x
                rel_y = v_y - ego_y
                distance = np.sqrt(rel_x**2 + rel_y**2)
                
                # Near miss detection
                if distance < self.near_miss_distance and abs(rel_y) < 2.0:
                    ep.near_miss_count += 1
                
                # TTC for vehicles ahead in same lane
                if 0 < rel_x < 50 and abs(rel_y) < 2.0:  # Ahead, approx same lane
                    relative_speed = ego_vx - v_vx
                    if relative_speed > 0.1:  # Approaching
                        ttc = rel_x / relative_speed
                        if ttc < ep.min_ttc:
                            ep.min_ttc = ttc
                
                # Improved overtake detection
                # Track the MAX relative position each vehicle had ahead of us.
                # If it was ahead and is now behind = overtake.
                # Works across ALL lanes.
                
                prev_data = ep._vehicle_positions.get(vid)
                
                if prev_data is None:
                    # First time seeing this vehicle
                    ep._vehicle_positions[vid] = {
                        'current': rel_x,
                        'max_ahead': rel_x if rel_x > 0 else 0,
                        'was_ahead': rel_x > self.overtake_threshold
                    }
                else:
                    was_ahead = prev_data['was_ahead']
                    max_ahead = prev_data['max_ahead']
                    
                    # Update max_ahead if vehicle moved further ahead
                    if rel_x > max_ahead:
                        max_ahead = rel_x
                    
                    # If not counted yet, was ahead, and now behind = OVERTAKE
                    if was_ahead and rel_x < -self.overtake_threshold:
                        if vid not in ep._overtaken_ids:
                            ep._overtaken_ids.add(vid)
                            ep.cars_overtaken += 1
                            
                            # Lane of the overtaken vehicle
                            v_lane = getattr(v, 'lane_index', None)
                            v_lane_str = f" (lane {v_lane[-1]})" if isinstance(v_lane, tuple) else ""
                            
                            if self.verbose:
                                print(f"  [Step {ep.steps}] OVERTAKE!{v_lane_str} max_ahead: {max_ahead:.1f}m -> now: {rel_x:.1f}m (Total: {ep.cars_overtaken})")
                    
                    # If vehicle goes ahead past threshold, mark it
                    if rel_x > self.overtake_threshold:
                        was_ahead = True
                    
                    ep._vehicle_positions[vid] = {
                        'current': rel_x,
                        'max_ahead': max_ahead,
                        'was_ahead': was_ahead
                    }
                
        except Exception as e:
            if self.verbose:
                print(f"  [Warning] Could not read env.unwrapped: {e}")
    
    def end_episode(self):
        """End the current episode and save data."""
        if self.current is not None:
            self.episodes.append(self.current)
            
            if self.verbose:
                ep = self.current
                status = "CRASH" if ep.crashed else ("TRUNCATED" if ep.truncated else "OK")
                print(f"Episode {len(self.episodes)}: {status} | "
                      f"Steps: {ep.steps} | Reward: {ep.total_reward:.1f} | "
                      f"Overtakes: {ep.cars_overtaken}")
            
            self.current = None
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all active metrics.
        
        Returns:
            Dictionary with metric values.
        """
        if not self.episodes:
            return {m: 0.0 for m in self.active_metrics}
        
        n = len(self.episodes)
        results = {}
        
        # Collision rate
        if 'collision_rate' in self.active_metrics:
            crashes = sum(1 for ep in self.episodes if ep.crashed)
            results['collision_rate'] = (crashes / n) * 100
        
        # Survival rate (inverse of collision rate)
        if 'survival_rate' in self.active_metrics:
            survived = sum(1 for ep in self.episodes if not ep.crashed)
            results['survival_rate'] = (survived / n) * 100
        
        # Average reward
        if 'avg_reward' in self.active_metrics:
            results['avg_reward'] = np.mean([ep.total_reward for ep in self.episodes])
        
        # Average speed
        if 'avg_speed' in self.active_metrics:
            all_speeds = [s for ep in self.episodes for s in ep.speeds]
            results['avg_speed'] = np.mean(all_speeds) if all_speeds else 0.0
        
        # Max speed
        if 'max_speed' in self.active_metrics:
            max_speeds = [max(ep.speeds) if ep.speeds else 0 for ep in self.episodes]
            results['max_speed'] = max(max_speeds) if max_speeds else 0.0
        
        # Cars overtaken (mean per episode)
        if 'cars_overtaken' in self.active_metrics:
            results['cars_overtaken'] = np.mean([ep.cars_overtaken for ep in self.episodes])
        
        # Total cars overtaken
        if 'total_cars_overtaken' in self.active_metrics:
            results['total_cars_overtaken'] = sum(ep.cars_overtaken for ep in self.episodes)
        
        # Average episode length
        if 'avg_episode_length' in self.active_metrics:
            results['avg_episode_length'] = np.mean([ep.steps for ep in self.episodes])
        
        # Lane changes (mean)
        if 'lane_changes' in self.active_metrics:
            results['lane_changes'] = np.mean([ep.lane_changes for ep in self.episodes])
        
        # Distance traveled (mean)
        if 'distance_traveled' in self.active_metrics:
            distances = []
            for ep in self.episodes:
                if ep.start_x is not None:
                    distances.append(ep.end_x - ep.start_x)
            results['distance_traveled'] = np.mean(distances) if distances else 0.0
        
        # Min TTC (mean of per-episode minimums, excluding inf)
        if 'min_ttc' in self.active_metrics:
            ttcs = [ep.min_ttc for ep in self.episodes if ep.min_ttc < float('inf')]
            results['min_ttc'] = np.mean(ttcs) if ttcs else float('inf')
        
        # Near miss rate
        if 'near_miss_rate' in self.active_metrics:
            near_misses = sum(1 for ep in self.episodes if ep.near_miss_count > 0)
            results['near_miss_rate'] = (near_misses / n) * 100
        
        return results
    
    def print_report(self, title: str = "Performance Report"):
        """Print a formatted metrics report."""
        metrics = self.compute()
        
        print(f"\n{'='*60}")
        print(f"{title:^60}")
        print(f"{'='*60}")
        print(f"Episodes evaluated: {len(self.episodes)}")
        print(f"{'-'*60}")
        
        # Custom order for readability
        order = [
            'collision_rate', 'survival_rate', 'avg_reward',
            'cars_overtaken', 'total_cars_overtaken',
            'avg_speed', 'max_speed', 'distance_traveled',
            'avg_episode_length', 'lane_changes',
            'min_ttc', 'near_miss_rate'
        ]
        
        for key in order:
            if key not in metrics:
                continue
            value = metrics[key]
            label = key.replace('_', ' ').title()
            
            # Type-specific formatting
            if 'rate' in key:
                print(f"  {label:.<40} {value:>12.1f}%")
            elif 'speed' in key:
                print(f"  {label:.<40} {value:>12.1f} m/s")
            elif key == 'distance_traveled':
                print(f"  {label:.<40} {value:>12.1f} m")
            elif key == 'min_ttc':
                if value == float('inf'):
                    print(f"  {label:.<40} {'N/A':>12}")
                else:
                    print(f"  {label:.<40} {value:>12.2f} s")
            elif 'total' in key:
                print(f"  {label:.<40} {int(value):>12}")
            else:
                print(f"  {label:.<40} {value:>12.2f}")
        
        print(f"{'='*60}\n")
    
    def save_json(self, filepath: str):
        """Save metrics to JSON format."""
        metrics = self.compute()
        
        # Handle inf for JSON
        for k, v in metrics.items():
            if v == float('inf'):
                metrics[k] = None
        
        data = {
            'timestamp': datetime.now().isoformat(),
            'n_episodes': len(self.episodes),
            'metrics': list(self.active_metrics),
            'results': metrics,
            'episodes': [
                {
                    'crashed': ep.crashed,
                    'steps': ep.steps,
                    'reward': ep.total_reward,
                    'cars_overtaken': ep.cars_overtaken,
                    'avg_speed': np.mean(ep.speeds) if ep.speeds else 0,
                }
                for ep in self.episodes
            ]
        }
        
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to: {path}")
        return path
    
    def reset(self):
        """Full tracker reset."""
        self.episodes.clear()
        self.current = None


def evaluate(
    model,
    env,
    n_episodes: int = 10,
    metrics: Optional[Set[str]] = None,
    render: bool = False,
    verbose: bool = False,
    seed: Optional[int] = None
) -> Dict[str, float]:
    """
    Evaluate an RL model on highway-env.
    
    Args:
        model: Model with .predict(obs) method
        env: Gymnasium environment
        n_episodes: Number of episodes
        metrics: Metrics to compute (None = all)
        render: Render the environment
        verbose: Print debug
    
    Returns:
        Dictionary with computed metrics.
    """
    tracker = HighwayMetrics(metrics=metrics, verbose=verbose)
    
    for ep_num in range(n_episodes):
        if seed is not None:
            obs, info = env.reset(seed=seed + ep_num)
        else:
            obs, info = env.reset()
            
        tracker.start_episode(env)
        
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            tracker.step(env, action, reward, done, truncated, info)
            
            if render:
                env.render()
        
        tracker.end_episode()
        
        # Progress
        if (ep_num + 1) % max(1, n_episodes // 10) == 0:
            print(f"Progress: {ep_num + 1}/{n_episodes} episodes")
    
    tracker.print_report()
    return tracker.compute()


# =============================================================================
#  CLI: CONFRONTO MODELLI
# =============================================================================

# Standard test scenarios for fair model comparison.
# Use the same reward/observation as ACCEL FIXED_PARAMS.
EVAL_FIXED_PARAMS = {
    'policy_frequency': 2,
    'collision_reward': -10.0,
    'high_speed_reward': 0.3,
    'right_lane_reward': 0.0,
    'lane_change_reward': 0.0,
    'reward_speed_range': [20, 30],
    'normalize_reward': False,
    'observation': {
        'type': 'Kinematics',
        'vehicles_count': 7,
        'features': ['presence', 'x', 'y', 'vx', 'vy'],
        'features_range': {
            'x': [-100, 100], 'y': [-100, 100],
            'vx': [-20, 20], 'vy': [-20, 20],
        },
        'absolute': False,
        'normalize': True,
        'see_behind': True,
        'order': 'sorted',
    },
}

EVAL_SCENARIOS = [
    {
        'name': 'Easy',
        'description': 'Stage 0 - 2 lanes, light traffic',
        'config': {
            **EVAL_FIXED_PARAMS,
            'lanes_count': 2, 'vehicles_count': 8,
            'vehicles_density': 0.8, 'duration': 30,
            'initial_spacing': 2.0,
            'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
        },
    },
    {
        'name': 'Baseline',
        'description': 'Stage 2 - 3 lanes, moderate traffic',
        'config': {
            **EVAL_FIXED_PARAMS,
            'lanes_count': 3, 'vehicles_count': 12,
            'vehicles_density': 0.8, 'duration': 40,
            'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
        },
    },
    {
        'name': 'Medium',
        'description': 'Stage 2 - 3 lanes, moderate traffic',
        'config': {
            **EVAL_FIXED_PARAMS,
            'lanes_count': 3, 'vehicles_count': 15,
            'vehicles_density': 1, 'duration': 40,
            'initial_spacing': 2.0,
            'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
        },
    },
    {
        'name': 'Hard',
        'description': 'Stage 4 - 3 lanes, dense traffic, long duration',
        'config': {
            **EVAL_FIXED_PARAMS,
            'lanes_count': 3, 'vehicles_count': 20,
            'vehicles_density': 1.2, 'duration': 50,
            'initial_spacing': 1.5,
            'other_vehicles_type': 'highway_env.vehicle.behavior.IDMVehicle',
        },
    },
    {
        'name': 'Expert',
        'description': 'Stage 6 - 4 lanes, dense, aggressive',
        'config': {
            **EVAL_FIXED_PARAMS,
            'lanes_count': 4, 'vehicles_count': 30,
            'vehicles_density': 1.5, 'duration': 60,
            'initial_spacing': 1.5,
            'other_vehicles_type': 'highway_env.vehicle.behavior.AggressiveVehicle',
        },
    },
]


def compare_models(
    models: Dict[str, str],
    scenarios: Optional[List[Dict]] = None,
    n_episodes: int = 10,
    seed: int = 42,
    device: str = 'auto',
    output_dir: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Compare one or more models on standard scenarios and save results.

    Args:
        models: {model_name: path_to_file.zip}
        scenarios: List of scenarios (default: EVAL_SCENARIOS)
        n_episodes: Episodes per scenario
        seed: Seed for reproducibility
        device: 'auto', 'cpu' or 'cuda'
        output_dir: Output folder (default: eval_results/<timestamp>)

    Returns:
        Nested dictionary: {model: {scenario: metrics}}
    """
    import gymnasium
    import highway_env  # Register highway-fast-v0, highway-v0, etc.
    import time

    try:
        from stable_baselines3 import DQN
    except ImportError:
        raise ImportError("stable_baselines3 not installed. Install with: pip install stable-baselines3")

    try:
        import torch
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
    except ImportError:
        device = 'cpu'

    if scenarios is None:
        scenarios = EVAL_SCENARIOS

    # Create output folder with timestamp
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"eval_results/compare_{timestamp}"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Dict] = {}

    # Metrics to track
    target_metrics = {
        'collision_rate', 'survival_rate',
        'cars_overtaken', 'avg_episode_length',
        'distance_traveled', 'avg_reward',
        'avg_speed', 'lane_changes',
    }

    for model_name, model_path in models.items():
        print(f"\n{'#'*65}")
        print(f"  MODEL: {model_name}")
        print(f"  Path:    {model_path}")
        print(f"{'#'*65}")

        model = DQN.load(model_path, device=device)
        model_results: Dict[str, Dict] = {}

        for sc in scenarios:
            sc_name = sc['name']
            sc_config = sc['config']

            print(f"\n  {'='*55}")
            print(f"  Scenario: {sc_name} — {sc.get('description', '')}")
            print(f"  Config: lanes={sc_config.get('lanes_count')}, "
                  f"vehicles={sc_config.get('vehicles_count')}, "
                  f"density={sc_config.get('vehicles_density', 1.0)}, "
                  f"duration={sc_config.get('duration')}")
            print(f"  {'='*55}")

            env = gymnasium.make("highway-fast-v0", config=sc_config)

            t0 = time.time()
            metrics = evaluate(
                model=model, env=env, n_episodes=n_episodes,
                metrics=target_metrics, render=False, verbose=False, seed=seed,
            )
            eval_time = time.time() - t0

            env.close()

            # Add extra info
            metrics['eval_time_seconds'] = round(eval_time, 2)
            model_results[sc_name] = metrics

        all_results[model_name] = model_results

        # --- Save JSON for this model ---
        model_json = {
            'model_name': model_name,
            'model_path': str(model_path),
            'timestamp': datetime.now().isoformat(),
            'n_episodes_per_scenario': n_episodes,
            'seed': seed,
            'device': device,
            'scenarios': {},
        }
        for sc in scenarios:
            sc_name = sc['name']
            sc_config = sc['config']
            sc_metrics = model_results.get(sc_name, {})

            # Filter config for readability (env params only, no observation blob)
            env_summary = {
                'lanes_count': sc_config.get('lanes_count'),
                'vehicles_count': sc_config.get('vehicles_count'),
                'vehicles_density': sc_config.get('vehicles_density'),
                'duration': sc_config.get('duration'),
                'initial_spacing': sc_config.get('initial_spacing'),
                'collision_reward': sc_config.get('collision_reward'),
                'high_speed_reward': sc_config.get('high_speed_reward'),
                'other_vehicles_type': sc_config.get('other_vehicles_type', '').split('.')[-1],
            }

            model_json['scenarios'][sc_name] = {
                'description': sc.get('description', ''),
                'environment': env_summary,
                'results': {
                    'collision_rate': round(sc_metrics.get('collision_rate', 0), 2),
                    'survival_rate': round(sc_metrics.get('survival_rate', 0), 2),
                    'cars_overtaken': round(sc_metrics.get('cars_overtaken', 0), 2),
                    'avg_episode_length': round(sc_metrics.get('avg_episode_length', 0), 1),
                    'distance_traveled': round(sc_metrics.get('distance_traveled', 0), 1),
                    'avg_reward': round(sc_metrics.get('avg_reward', 0), 2),
                    'avg_speed': round(sc_metrics.get('avg_speed', 0), 2),
                    'lane_changes': round(sc_metrics.get('lane_changes', 0), 2),
                    'eval_time_seconds': sc_metrics.get('eval_time_seconds', 0),
                },
            }

        json_path = out_path / f"{model_name}.json"
        with open(json_path, 'w') as f:
            json.dump(model_json, f, indent=2)
        print(f"\n  [SAVED] {json_path}")

    # --- Tabella comparativa in console ---
    _print_comparison_table(all_results, scenarios)

    # --- Save comparison summary ---
    summary = {
        'timestamp': datetime.now().isoformat(),
        'models': list(models.keys()),
        'n_episodes_per_scenario': n_episodes,
        'results': {},
    }
    for sc in scenarios:
        sc_name = sc['name']
        summary['results'][sc_name] = {}
        for m_name in models:
            m_res = all_results.get(m_name, {}).get(sc_name, {})
            summary['results'][sc_name][m_name] = {
                'survival_rate': round(m_res.get('survival_rate', 0), 1),
                'avg_reward': round(m_res.get('avg_reward', 0), 2),
                'cars_overtaken': round(m_res.get('cars_overtaken', 0), 2),
                'distance_traveled': round(m_res.get('distance_traveled', 0), 1),
            }
    with open(out_path / "comparison_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {out_path}")
    return all_results


def _print_comparison_table(
    all_results: Dict[str, Dict],
    scenarios: List[Dict],
):
    """Print a readable comparison table for all models."""
    model_names = list(all_results.keys())
    if not model_names:
        return

    # Column widths
    name_w = max(12, max(len(n) for n in model_names) + 2)
    col_w = name_w

    print(f"\n{'='*75}")
    print(f"{'MODEL COMPARISON':^75}")
    print(f"{'='*75}")

    for sc in scenarios:
        sc_name = sc['name']
        print(f"\n  --- {sc_name}: {sc.get('description', '')} ---")

        # Header
        header = f"  {'Metric':<22}"
        for m in model_names:
            header += f" {m:>{col_w}}"
        print(header)
        print(f"  {'-'*22}" + f" {'-'*col_w}" * len(model_names))

        rows = [
            ('Survival %',       'survival_rate',       '{:.1f}%'),
            ('Collision %',      'collision_rate',      '{:.1f}%'),
            ('Avg Reward',       'avg_reward',          '{:.2f}'),
            ('Cars Overtaken',   'cars_overtaken',      '{:.1f}'),
            ('Distance (m)',     'distance_traveled',   '{:.0f}'),
            ('Length (steps)',   'avg_episode_length',  '{:.0f}'),
            ('Speed (m/s)',      'avg_speed',           '{:.1f}'),
            ('Lane Changes',    'lane_changes',        '{:.1f}'),
        ]

        for label, key, fmt in rows:
            line = f"  {label:<22}"
            for m in model_names:
                val = all_results.get(m, {}).get(sc_name, {}).get(key, 0)
                line += f" {fmt.format(val):>{col_w}}"
            print(line)

    print(f"\n{'='*75}\n")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Compare RL models on standard highway-env scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two models
  python metrics_tracker.py --dqn_baseline ./highway_dqn/best_model.zip \\
                            --dqn_accel ./highway_dqn_accel/dqn_accel_final.zip

  # Single model, 20 episodes
  python metrics_tracker.py --my_model ./path/model.zip --episodes 20

  # Three models with custom output
  python metrics_tracker.py --baseline model_a.zip --accel model_b.zip \\
                            --curriculum model_c.zip --output ./my_eval
        """
    )

    parser.add_argument('--episodes', type=int, default=10,
                        help='Episodes per scenario (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed for reproducibility (default: 42)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device (default: auto)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output folder (default: eval_results/compare_<timestamp>)')

    # Two-phase parsing: known args first, then dynamic model args
    args, remaining = parser.parse_known_args()

    # Parse model arguments: --model_name /path/to/model.zip
    models: Dict[str, str] = {}
    i = 0
    while i < len(remaining):
        token = remaining[i]
        if token.startswith('--') and i + 1 < len(remaining):
            name = token.lstrip('-').replace('-', '_')
            path = remaining[i + 1]
            if not Path(path).exists():
                print(f"[ERROR] File not found: {path}")
                sys.exit(1)
            models[name] = path
            i += 2
        else:
            print(f"[ERROR] Unrecognized argument: {token}")
            print("Usage: --model_name /path/to/model.zip")
            sys.exit(1)

    if not models:
        print("[ERROR] Specify at least one model.")
        print("Usage: python metrics_tracker.py --model_name /path/model.zip")
        print("       python metrics_tracker.py --baseline a.zip --accel b.zip")
        sys.exit(1)

    print(f"\nModels to compare: {len(models)}")
    for name, path in models.items():
        print(f"  - {name}: {path}")
    print(f"Episodes per scenario: {args.episodes}")
    print(f"Seed: {args.seed}")

    compare_models(
        models=models,
        n_episodes=args.episodes,
        seed=args.seed,
        device=args.device,
        output_dir=args.output,
    )
