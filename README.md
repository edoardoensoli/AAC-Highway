# Curriculum Learning for Highway-Env: Implementation Guide

This repository contains implementations of three curriculum learning approaches for training robust RL agents in highway-env:

1. **Baseline** - Single fixed environment
2. **Domain Randomization (DR)** - Random environment sampling
3. **Prioritized Level Replay (PLR)** - Intelligent curriculum learning

## 📚 Academic Background

### Key Papers

**Domain Randomization:**
- Tobin et al. (2017) - "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World"
  - arXiv: https://arxiv.org/abs/1703.06907
  
**Prioritized Level Replay:**
- Jiang et al. (2021) - "Prioritized Level Replay"
  - arXiv: https://arxiv.org/abs/2010.03934
  - Code: https://github.com/facebookresearch/level-replay
  
**ACCEL (Advanced):**
- Parker-Holder et al. (2022) - "Evolving Curricula with Regret-Based Environment Design"
  - arXiv: https://arxiv.org/abs/2203.01302
  - Website: https://accelagent.github.io/

See `curriculum_learning_guide.md` for comprehensive theoretical background and implementation details.

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install gymnasium highway-env stable-baselines3 torch tqdm

# Install highway-env if not already installed
pip install highway-env
```

### Basic Usage

#### 1. Train Baseline (Single Environment)

```bash
python train_with_curriculum.py --method baseline --total-timesteps 100000
```

#### 2. Train with Domain Randomization

```bash
python train_with_curriculum.py --method dr --total-timesteps 100000
```

#### 3. Train with PLR (Recommended)

```bash
python train_with_curriculum.py --method plr --total-timesteps 100000
```

### Command Line Options

```
--method: baseline | dr | plr (default: plr)
--total-timesteps: Total training steps (default: 100000)
--log-dir: Directory for logs (default: highway_curriculum_logs)
--seed: Random seed (default: 42)
```

## 📊 Monitoring Training

### TensorBoard

All methods log to TensorBoard:

```bash
tensorboard --logdir highway_curriculum_logs_<timestamp>
```

Key metrics to watch:
- `train/ep_rew_mean` - Training reward
- `eval/mean_reward` - Test set generalization
- `plr/num_seen_levels` - How many environments explored
- `plr/avg_score` - Average difficulty of curriculum

### Expected Results

After 100k steps:
- **Baseline**: Good on training env, poor generalization
- **DR**: Moderate performance, okay generalization
- **PLR**: Best generalization to test environments

## 🔧 Customization

### Modify Environment Configurations

Edit `plr_implementation.py`, function `_generate_default_configs()`:

```python
def _generate_default_configs(self):
    configs = []
    
    # Add your custom parameter ranges
    for lanes in [2, 3, 4, 5]:  # More lanes
        for density in [0.5, 1.0, 1.5, 2.0, 2.5]:  # Wider density range
            for vehicles in range(10, 50, 5):  # More vehicle counts
                config = {
                    'lanes_count': lanes,
                    'vehicles_density': density,
                    'vehicles_count': vehicles,
                    'duration': 60,
                    # Add more parameters
                    'collision_reward': -5.0,
                    'high_speed_reward': 0.4,
                }
                configs.append(config)
    
    return configs
```

### Tune PLR Hyperparameters

In `train_with_curriculum.py`, modify the PLR initialization:

```python
plr = PLRManager(
    env_id="highway-v0",
    train_env_configs=train_configs,
    score_function='value_loss',  # or 'advantage', 'return'
    replay_probability=0.8,  # 0.7-0.9, higher = more curriculum
    temperature=0.1,  # 0.05-0.2, lower = more greedy
    staleness_coef=0.1,  # 0.05-0.2, higher = more exploration
    buffer_size=50,  # 50-200, number of levels to track
)
```

### Change RL Algorithm

Replace DQN with PPO or other SB3 algorithms:

```python
from stable_baselines3 import PPO

model = PPO(
    'MlpPolicy',
    env,
    policy_kwargs=dict(net_arch=[256, 256]),
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    # ... other PPO params
)
```

## 📈 Evaluation

### Test Generalization

```python
import gymnasium
from stable_baselines3 import DQN
import json
import numpy as np

# Load model
model = DQN.load("highway_curriculum_logs_<timestamp>/plr/model")

# Load test configs
with open("highway_curriculum_logs_<timestamp>/plr/test_configs.json") as f:
    test_configs = json.load(f)

# Evaluate
results = []
for config in test_configs:
    env = gymnasium.make("highway-v0", config=config)
    
    # Run 10 episodes
    rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        done = truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        rewards.append(episode_reward)
    
    results.append({
        'config': config,
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
    })

# Print results
for i, r in enumerate(results):
    print(f"Config {i}: {r['mean_reward']:.2f} +/- {r['std_reward']:.2f}")
    print(f"  {r['config']}")

print(f"\nOverall mean: {np.mean([r['mean_reward'] for r in results]):.2f}")
```

### Visualize Learned Policy

```python
import gymnasium
from stable_baselines3 import DQN

# Load model
model = DQN.load("highway_curriculum_logs_<timestamp>/plr/model")

# Create environment with rendering
env = gymnasium.make(
    "highway-v0",
    config={'lanes_count': 4, 'vehicles_count': 30, 'vehicles_density': 1.5},
    render_mode='human'
)

# Run episode
obs, _ = env.reset()
done = truncated = False

while not (done or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()

env.close()
```

## 🎯 Next Steps: ACCEL Implementation

After mastering PLR, implement ACCEL for even better generalization:

### Key Components Needed:

1. **Mutation Function**: Define how to mutate environments
```python
def mutate_config(config, mutation_rate=0.1):
    new_config = copy.deepcopy(config)
    
    if np.random.random() < mutation_rate:
        # Mutate lanes
        new_config['lanes_count'] += np.random.choice([-1, 0, 1])
        new_config['lanes_count'] = np.clip(new_config['lanes_count'], 2, 5)
    
    # ... more mutations
    return new_config
```

2. **Regret Estimation**: Measure learning potential
```python
def estimate_regret(episode_data):
    # High positive advantage = room for improvement
    advantages = episode_data['advantages']
    return np.mean(np.maximum(advantages, 0))
```

3. **Evolution Loop**: Sample → Mutate → Evaluate → Update

See `curriculum_learning_guide.md` for full ACCEL implementation.

## 📁 File Structure

```
.
├── curriculum_learning_guide.md    # Comprehensive theory & implementation guide
├── plr_implementation.py           # PLR core implementation
├── train_with_curriculum.py        # Main training script
├── README.md                        # This file
│
├── highway_curriculum_logs_<timestamp>/
│   ├── baseline/
│   │   └── model.zip
│   ├── dr/
│   │   └── model.zip
│   └── plr/
│       ├── model.zip
│       ├── plr_state.pkl
│       ├── train_configs.json
│       └── test_configs.json
```

## 🐛 Troubleshooting

### Issue: Training is slow
- **Solution**: Reduce `total_timesteps` or use a smaller environment configuration set
- **Solution**: Use GPU if available (automatically detected)

### Issue: PLR not showing improvement over DR
- **Solution**: Increase training time (PLR needs time to build curriculum)
- **Solution**: Tune hyperparameters (see Customization section)
- **Solution**: Ensure sufficient environment diversity

### Issue: Agent not generalizing to test set
- **Solution**: Increase environment diversity in training set
- **Solution**: Make sure test set is truly held-out (not overlapping with train)
- **Solution**: Try different `score_function` in PLR ('value_loss', 'advantage', 'return')

### Issue: "CUDA out of memory"
- **Solution**: Reduce `batch_size` in model configuration
- **Solution**: Use CPU instead: model will automatically detect and use CPU if CUDA unavailable

## 📖 Learning Resources

### Video Tutorials
- Yannic Kilcher's "ACCEL - Evolving Curricula with Regret-Based Environment Design" on YouTube
- "Prioritized Level Replay" presentation at ICML 2021

### Additional Papers
- **Survey**: Narvekar et al. (2020) "Curriculum Learning for Reinforcement Learning Domains: A Framework and Survey"
  - URL: https://jmlr.org/papers/volume21/20-212/20-212.pdf

- **Sim2Real**: Understanding Domain Randomization (Xiong et al., ICLR 2022)
  - URL: https://openreview.net/pdf?id=T8vZHIRTrY

### Related Repositories
- Official PLR: https://github.com/facebookresearch/level-replay
- ACCEL website: https://accelagent.github.io/
- Highway-Env: https://github.com/Farama-Foundation/HighwayEnv

## 💡 Tips for Success

1. **Start Simple**: Begin with Domain Randomization to understand your environment space
2. **Monitor Carefully**: Watch both training and test performance
3. **Be Patient**: PLR needs 50k+ steps to show clear benefits
4. **Tune Gradually**: Change one hyperparameter at a time
5. **Evaluate Properly**: Always use held-out test configurations

## 🤝 Contributing

Found a bug or want to improve the implementation? Here's how:

1. Test on your environment variations
2. Compare against baseline
3. Share results and insights

## 📝 Citation

If you use this code in your research, please cite the original papers:

```bibtex
@inproceedings{jiang2021prioritized,
  title={Prioritized Level Replay},
  author={Jiang, Minqi and Grefenstette, Edward and Rockt{\"a}schel, Tim},
  booktitle={ICML},
  year={2021}
}

@inproceedings{parkerholder2022accel,
  title={Evolving Curricula with Regret-Based Environment Design},
  author={Parker-Holder, Jack and Jiang, Minqi and Dennis, Michael and Samvelyan, Mikayel and Foerster, Jakob and Grefenstette, Edward and Rockt{\"a}schel, Tim},
  booktitle={ICML},
  year={2022}
}

@article{tobin2017domain,
  title={Domain randomization for transferring deep neural networks from simulation to the real world},
  author={Tobin, Josh and Fong, Rachel and Ray, Alex and Schneider, Jonas and Zaremba, Wojciech and Abbeel, Pieter},
  journal={IROS},
  year={2017}
}
```

## 📧 Support

Questions? Check:
1. `curriculum_learning_guide.md` for detailed explanations
2. GitHub Issues for common problems
3. Original paper repositories for reference implementations

---

**Good luck with your curriculum learning experiments! 🎓🚗**
