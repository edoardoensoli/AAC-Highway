# PPO + ACCEL Highway Training

## Train

```bash
# Default training (500k steps, 8 parallel envs)
python src/ppo_accel/train.py train

# Custom timesteps and envs
python src/ppo_accel/train.py train --timesteps 1000000 --num-envs 12

# Resume from pretrained model
python src/ppo_accel/train.py train --pretrained ./ppo_accel_out/best_model.zip --timesteps 500000

# Custom save directory
python src/ppo_accel/train.py train --save-dir ./my_run

# Tune PLR/ACCEL
python src/ppo_accel/train.py train --plr-buffer 1000 --replay-prob 0.9 --editor-prob 0.4 --num-edits 4

# Tune PPO
python src/ppo_accel/train.py train --lr 1e-4 --n-steps 512 --batch-size 128 --n-epochs 15
```

## Evaluate

```bash
python src/ppo_accel/train.py eval ./ppo_accel_out/best_model.zip
python src/ppo_accel/train.py eval ./ppo_accel_out/ppo_accel_final.zip --eval-episodes 100
```

## Output files

| File | Description |
|------|-------------|
| `best_model.zip` | Best model (highest avg reward over 50 ep window) |
| `ppo_accel_final.zip` | Model at end of training |
| `checkpoint_latest.zip` | Latest periodic checkpoint |
| `plr_latest.pkl` | PLR buffer state |
| `training_config.json` | All hyperparameters |
| `training_stats.json` | Final training stats |
| `best_model_info.json` | Step/reward at best save |
| `tensorboard/` | TensorBoard logs |

## Architecture

```
src/ppo_accel/
  env.py       - HighwayEnvWrapper (lazy config, fixed rewards)
  plr.py       - LevelSampler (PLR buffer, rank scoring, staleness)
  accel.py     - ACCELGenerator (mutations, random levels, difficulty)
  callback.py  - ACCELCallback (SB3 loop: score -> decide -> apply)
  train.py     - CLI entry point (train / eval)
```
