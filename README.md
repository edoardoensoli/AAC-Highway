# AAC-Highway

DQN agents for autonomous highway driving using [highway-env](https://github.com/Farama-Foundation/HighwayEnv).
Two training approaches are compared:

- **DQN Baseline** -- standard DQN on a fixed environment configuration.
- **DQN + ACCEL** -- DQN with curriculum learning (ACCEL: Evolving Curricula with Regret-Based Environment Design, Parker-Holder et al. 2022). The agent progresses through 7 difficulty stages, from easy traffic to dense aggressive scenarios.

Both models share the same reward function, observation space (Kinematics 7x5), and network architecture, so results are directly comparable.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
```

## Training

### DQN Baseline

Edit the constants at the top of `src/dqn_baseline.py` to configure training:


| Variable          | Default     | Description                                            |
| ----------------- | ----------- | ------------------------------------------------------ |
| `TRAIN`           | `True`      | Set to`False` to skip training and run evaluation only |
| `TOTAL_TIMESTEPS` | `1_000_000` | Total training steps                                   |
| `NUM_ENVS`        | `4`         | Parallel environments (SubprocVecEnv)                  |

```bash
python src/dqn_baseline.py
```

Models are saved to `highway_dqn/`.

### DQN + ACCEL

```bash
python src/dqn_accel.py --timesteps 1000000
```

Key CLI arguments:


| Argument              | Default              | Description                       |
| --------------------- | -------------------- | --------------------------------- |
| `--pretrained`        | None                 | Path to pre-trained model (.zip)  |
| `--timesteps`         | 500000               | Total training steps              |
| `--num-envs`          | 8                    | Parallel environments             |
| `--mastery-threshold` | 0.80                 | Survival rate to advance stage    |
| `--start-stage`       | 0                    | Starting curriculum stage (0-6)   |
| `--lr`                | 5e-4                 | Learning rate                     |
| `--batch-size`        | 64                   | Batch size                        |
| `--gamma`             | 0.95                 | Discount factor                   |
| `--save-dir`          | `./dqn_accel_models` | Output directory                  |
| `--no-accel`          | --                   | Disable mutations (PLR only)      |
| `--eval-only`         | None                 | Evaluate a model without training |
| `--config`            | None                 | JSON config file (overrides CLI)  |

Example with pre-trained baseline:

```bash
python src/dqn_accel.py --pretrained highway_dqn/dqn_baseline_1M.zip --timesteps 1000000
```

Evaluate a trained model:

```bash
python src/dqn_accel.py --eval-only highway_dqn_accel/dqn_accel_final_1M.zip
```

Models are saved to `highway_dqn_accel/`.

## Evaluation

### Single model (with rendering)

```bash
python test_render.py --model highway_dqn_accel/dqn_accel_final_1M.zip --difficulty medium
python test_render.py --model highway_dqn_accel/dqn_accel_final_1M.zip --all-difficulties
```

Difficulties: `easy`, `medium`, `hard`, `expert`.

### Interactive viewer (GUI)

```bash
python src/interactive_viewer.py
```

Adjustable parameters via sliders and buttons: lanes, vehicles, density, FPS, vehicle aggressiveness, model selection. Controls: `SPACE` pause/resume, `ESC` quit.

### Model comparison

Compare any number of models across standardized scenarios:

```bash
python src/metrics_tracker.py \
    --dqn_baseline highway_dqn/dqn_baseline_1M.zip \
    --dqn_accel highway_dqn_accel/dqn_accel_final_1M.zip \
    --episodes 50 \
    --output ./eval_results
```

Arguments:


| Argument         | Default | Description                        |
| ---------------- | ------- | ---------------------------------- |
| `--<model_name>` | --      | One or more`--name path.zip` pairs |
| `--episodes`     | 10      | Episodes per scenario              |
| `--seed`         | 42      | Random seed                        |
| `--output`       | auto    | Output folder for JSON results     |

## Pre-trained models

### highway_dqn/ (baseline)


| File                    | Steps |
| ----------------------- | ----- |
| `dqn_baseline_200k.zip` | 200k  |
| `dqn_baseline_500k.zip` | 500k  |
| `dqn_baseline_1M.zip`   | 1M    |

### highway_dqn_accel/ (ACCEL curriculum)


| File                       | Steps |
| -------------------------- | ----- |
| `dqn_accel_final_200k.zip` | 200k  |
| `dqn_accel_final_500k.zip` | 500k  |
| `dqn_accel_final_1M.zip`   | 1M    |

Intermediate checkpoints are also available in each folder (`checkpoint_step*.zip`, `dqn_checkpoint_*_steps.zip`).

## Project structure

```
src/
    dqn_baseline.py          # Baseline DQN training
    dqn_accel.py             # DQN + ACCEL curriculum training
    metrics_tracker.py       # Model comparison across scenarios
    aggressive_vehicle.py    # Custom aggressive IDM vehicle
    interactive_viewer.py    # Interactive GUI viewer
    test_render.py           # CLI model evaluation with rendering
requirements.txt
```

## References

- Parker-Holder et al. (2022) "Evolving Curricula with Regret-Based Environment Design" -- [arXiv:2203.01302](https://arxiv.org/abs/2203.01302)
- Jiang et al. (2021) "Prioritized Level Replay" -- [arXiv:2010.03934](https://arxiv.org/abs/2010.03934)
- highway-env -- [github.com/Farama-Foundation/HighwayEnv](https://github.com/Farama-Foundation/HighwayEnv)
