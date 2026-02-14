#!/usr/bin/env python3
"""
Run metrics_tracker.py for 3 checkpoints (200k, 500k, 1M) sequentially.

Usage: python run_metrics_batch.py [--episodes N]

Creates output folders `test_200k/`, `test_500k/`, `test_1M/` and writes logs there.
"""
import subprocess
import sys
import os
from pathlib import Path
import argparse
import time

ROOT = Path(__file__).resolve().parent
PY = sys.executable or "python"

MODEL_TEMPLATES = {
    'dqn_plr': 'highway_dqn_plr/dqn_plr_{s}.zip',
    'dqn_baseline': 'highway_dqn/dqn_baseline_{s}.zip',
    'dqn_accel': 'highway_dqn_accel/dqn_accel_final_{s}.zip',
    'ppo': 'highway_ppo/ppo_baseline_{s}.zip',
}

DEFAULT_RUNS = ['200k', '500k', '1M']


def build_command(suffix: str, output_dir: str, episodes: int) -> list:
    cmd = [PY, str(ROOT / 'src' / 'metrics_tracker.py')]
    cmd += ['--dqn_plr', MODEL_TEMPLATES['dqn_plr'].format(s=suffix)]
    cmd += ['--dqn_baseline', MODEL_TEMPLATES['dqn_baseline'].format(s=suffix)]
    cmd += ['--dqn_accel', MODEL_TEMPLATES['dqn_accel'].format(s=suffix)]
    cmd += ['--ppo', MODEL_TEMPLATES['ppo'].format(s=suffix)]
    cmd += ['--output', output_dir]
    cmd += ['--episodes', str(episodes)]
    return cmd


def run_batch(suffix: str, episodes: int):
    output_dir = ROOT / f'test_{suffix}'
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_command(suffix, str(output_dir), episodes)

    log_path = output_dir / f'run_{suffix}.log'

    print(f"\n=== RUN {suffix} ===")
    print("Command:", ' '.join(cmd))
    print(f"Log: {log_path}\n")

    start = time.time()
    with open(log_path, 'wb') as lf:
        # Run the command, stream stdout/stderr to log file
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
        try:
            p.wait()
        except KeyboardInterrupt:
            p.terminate()
            p.wait()
            print("Run interrupted by user")
            return False

    elapsed = time.time() - start
    if p.returncode == 0:
        print(f"Completed {suffix} in {elapsed:.1f}s — returncode 0")
        return True
    else:
        print(f"Run {suffix} failed (returncode={p.returncode}). See log: {log_path}")
        return False


def main(runs, episodes):
    for s in runs:
        ok = run_batch(s, episodes)
        if not ok:
            print(f"Warning: run {s} failed — continuing to next run")
    print("\nAll runs finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run metrics_tracker batches for 200k, 500k, 1M')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per scenario (default: 100)')
    parser.add_argument('--runs', nargs='+', default=DEFAULT_RUNS, help='Suffixes to run (default: 200k 500k 1M)')
    args = parser.parse_args()

    main(args.runs, args.episodes)
