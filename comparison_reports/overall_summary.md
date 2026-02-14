# Overall Comparison: survival and avg_reward (200k / 500k / 1M)

## Scenario: Baseline

| Model | Survival (200k) | AvgReward (200k) | Survival (500k) | AvgReward (500k) | Survival (1M) | AvgReward (1M) |
|---|---:|---:|---:|---:|---:|---:|
| dqn_accel | 51.0 | 9.74 | 75.0 | 17.96 | 87.0 | 20.62 |
| dqn_baseline | 22.0 | 6.01 | 68.0 | 17.61 | 82.0 | 20.11 |
| dqn_plr | 0.0 | -4.45 | 0.0 | -3.44 | 43.0 | -1.06 |
| ppo | 78.0 | 3.42 | 81.0 | 10.13 | 92.0 | 15.1 |

## Scenario: Easy

| Model | Survival (200k) | AvgReward (200k) | Survival (500k) | AvgReward (500k) | Survival (1M) | AvgReward (1M) |
|---|---:|---:|---:|---:|---:|---:|
| dqn_accel | 84.0 | 11.73 | 60.0 | 9.54 | 83.0 | 12.32 |
| dqn_baseline | 32.0 | 3.79 | 28.0 | 4.03 | 54.0 | 9.11 |
| dqn_plr | 1.0 | -6.09 | 1.0 | -5.19 | 57.0 | -2.38 |
| ppo | 41.0 | -1.83 | 36.0 | 0.23 | 80.0 | 10.68 |

## Scenario: Expert

| Model | Survival (200k) | AvgReward (200k) | Survival (500k) | AvgReward (500k) | Survival (1M) | AvgReward (1M) |
|---|---:|---:|---:|---:|---:|---:|
| dqn_accel | 0.0 | -6.91 | 4.0 | -3.84 | 3.0 | -3.17 |
| dqn_baseline | 0.0 | -7.3 | 0.0 | -5.79 | 6.0 | -3.37 |
| dqn_plr | 0.0 | -7.78 | 0.0 | -7.79 | 3.0 | -8.19 |
| ppo | 24.0 | -5.4 | 29.0 | -4.02 | 13.0 | -4.54 |

## Scenario: Hard

| Model | Survival (200k) | AvgReward (200k) | Survival (500k) | AvgReward (500k) | Survival (1M) | AvgReward (1M) |
|---|---:|---:|---:|---:|---:|---:|
| dqn_accel | 0.0 | -4.43 | 25.0 | 5.72 | 31.0 | 9.06 |
| dqn_baseline | 0.0 | -5.08 | 22.0 | 5.27 | 26.0 | 7.26 |
| dqn_plr | 0.0 | -6.96 | 0.0 | -6.51 | 35.0 | -4.54 |
| ppo | 45.0 | -3.07 | 28.0 | -2.67 | 19.0 | -1.63 |

## Scenario: Medium

| Model | Survival (200k) | AvgReward (200k) | Survival (500k) | AvgReward (500k) | Survival (1M) | AvgReward (1M) |
|---|---:|---:|---:|---:|---:|---:|
| dqn_accel | 17.0 | 0.67 | 51.0 | 12.28 | 67.0 | 16.01 |
| dqn_baseline | 5.0 | -0.9 | 52.0 | 12.83 | 64.0 | 15.29 |
| dqn_plr | 0.0 | -5.62 | 0.0 | -5.13 | 49.0 | -2.32 |
| ppo | 72.0 | 0.74 | 62.0 | 3.2 | 77.0 | 8.33 |

