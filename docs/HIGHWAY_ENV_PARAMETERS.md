# Highway-Env Parameters Reference Guide

A comprehensive guide to all configurable parameters in the highway-env environment for training and testing autonomous driving agents.

---

## Table of Contents

1. [Environment Configuration](#environment-configuration)
2. [Vehicle Behavior Types](#vehicle-behavior-types)
3. [IDMVehicle Parameters](#idmvehicle-parameters)
4. [AggressiveVehicle Parameters](#aggressivevehicle-parameters)
5. [DefensiveVehicle Parameters](#defensivevehicle-parameters)
6. [LinearVehicle Parameters](#linearvehicle-parameters)
7. [Difficulty Tuning Guide](#difficulty-tuning-guide)
8. [Custom Vehicle Classes](#custom-vehicle-classes)

---

## Environment Configuration

These parameters are passed to `gymnasium.make()` in the `config` dictionary.

### Road Parameters

| Parameter         | Type  | Default | Range | Description                                                                                      |
| ----------------- | ----- | ------- | ----- | ------------------------------------------------------------------------------------------------ |
| `lanes_count`     | int   | 4       | 1-8   | Number of lanes on the highway. More lanes = more escape routes but also more traffic complexity |
| `initial_spacing` | float | 2       | 0.5-5 | Initial spacing factor between vehicles. Lower = denser initial placement                        |

### Traffic Parameters

| Parameter             | Type   | Default                                     | Range     | Description                                                    |
| --------------------- | ------ | ------------------------------------------- | --------- | -------------------------------------------------------------- |
| `vehicles_count`      | int    | 50                                          | 1-100+    | Total number of other vehicles on the road                     |
| `vehicles_density`    | float  | 1                                           | 0.1-3.0   | Density multiplier for vehicle spawning. Higher = more crowded |
| `other_vehicles_type` | string | `"highway_env.vehicle.behavior.IDMVehicle"` | See below | Class path for other vehicles' behavior                        |
| `controlled_vehicles` | int    | 1                                           | 1-4       | Number of ego vehicles (for multi-agent)                       |

### Simulation Parameters

| Parameter              | Type  | Default | Range  | Description                                                           |
| ---------------------- | ----- | ------- | ------ | --------------------------------------------------------------------- |
| `duration`             | float | 40      | 10-300 | Episode duration in seconds                                           |
| `simulation_frequency` | int   | 15      | 5-60   | Physics simulation frequency in Hz. Higher = more accurate but slower |
| `policy_frequency`     | int   | 1       | 1-15   | Decision frequency in Hz. How often the agent chooses an action       |

### Reward Parameters

| Parameter            | Type  | Default  | Range          | Description                                         |
| -------------------- | ----- | -------- | -------------- | --------------------------------------------------- |
| `collision_reward`   | float | -1       | -10 to 0       | Reward (penalty) for colliding with another vehicle |
| `reward_speed_range` | list  | [20, 30] | [min, max] m/s | Speed range for high-speed reward mapping           |
| `right_lane_reward`  | float | 0.1      | 0-1            | Reward for driving on the rightmost lane            |
| `high_speed_reward`  | float | 0.4      | 0-1            | Reward for maintaining high speed                   |
| `lane_change_reward` | float | 0        | -0.1 to 0.1    | Reward/penalty for lane changes                     |

### Display Parameters

| Parameter            | Type  | Default    | Description                      |
| -------------------- | ----- | ---------- | -------------------------------- |
| `screen_width`       | int   | 600        | Render window width in pixels    |
| `screen_height`      | int   | 150        | Render window height in pixels   |
| `centering_position` | list  | [0.3, 0.5] | Camera centering position [x, y] |
| `scaling`            | float | 5.5        | Zoom level for rendering         |
| `show_trajectories`  | bool  | False      | Display vehicle trajectories     |
| `render_agent`       | bool  | True       | Render the ego vehicle           |

---

## Vehicle Behavior Types

Available options for `other_vehicles_type`:

| Type                  | Class Path                                       | Description                                                         |
| --------------------- | ------------------------------------------------ | ------------------------------------------------------------------- |
| **IDMVehicle**        | `highway_env.vehicle.behavior.IDMVehicle`        | Standard Intelligent Driver Model - realistic, balanced behavior    |
| **AggressiveVehicle** | `highway_env.vehicle.behavior.AggressiveVehicle` | Aggressive driving - frequent lane changes, less following distance |
| **DefensiveVehicle**  | `highway_env.vehicle.behavior.DefensiveVehicle`  | Defensive driving - larger gaps, fewer lane changes                 |
| **LinearVehicle**     | `highway_env.vehicle.behavior.LinearVehicle`     | Linear controller - simpler, more predictable                       |

---

## IDMVehicle Parameters

The Intelligent Driver Model (IDM) vehicle is the default and most commonly used. It combines:

- **Longitudinal control**: IDM acceleration model
- **Lateral control**: MOBIL lane-change model

### Longitudinal (IDM) Parameters

| Parameter         | Default    | Range     | Unit | Effect                                          |
| ----------------- | ---------- | --------- | ---- | ----------------------------------------------- |
| `ACC_MAX`         | 6.0        | 3-12      | m/s² | Maximum possible acceleration                   |
| `COMFORT_ACC_MAX` | 3.0        | 1-6       | m/s² | Desired comfortable acceleration                |
| `COMFORT_ACC_MIN` | -5.0       | -10 to -2 | m/s² | Desired comfortable deceleration (braking)      |
| `DISTANCE_WANTED` | ~10        | 3-20      | m    | Minimum jam distance to front vehicle           |
| `TIME_WANTED`     | 1.5        | 0.5-3.0   | s    | Desired time headway gap                        |
| `DELTA`           | 4.0        | 2-6       | -    | Velocity exponent (acceleration aggressiveness) |
| `DELTA_RANGE`     | [3.5, 4.5] | -         | -    | Range for random delta selection                |

#### IDM Equation:

```
ẍ = a * [1 - (v/v₀)^δ - (d*/d)²]

where:
  d* = d₀ + T*v + v*Δv/(2*√(a*b))
```

### Lateral (MOBIL) Parameters

| Parameter                         | Default | Range | Unit | Effect                                                     |
| --------------------------------- | ------- | ----- | ---- | ---------------------------------------------------------- |
| `POLITENESS`                      | 0.0     | 0-1   | -    | Consideration for other vehicles (0=selfish, 1=altruistic) |
| `LANE_CHANGE_MIN_ACC_GAIN`        | 0.1     | 0-1   | m/s² | Minimum acceleration advantage to trigger lane change      |
| `LANE_CHANGE_MAX_BRAKING_IMPOSED` | 2.0     | 1-5   | m/s² | Maximum braking imposed on new follower during cut-in      |
| `LANE_CHANGE_DELAY`               | 1.0     | 0.5-3 | s    | Minimum time between lane change decisions                 |

#### MOBIL Lane Change Conditions:

1. **Safety**: New follower's deceleration < `LANE_CHANGE_MAX_BRAKING_IMPOSED`
2. **Incentive**: Acceleration gain > `LANE_CHANGE_MIN_ACC_GAIN`

---

## AggressiveVehicle Parameters

Inherits from `LinearVehicle` with aggressive parameter tuning:

| Parameter                  | Value | Comparison to IDM               |
| -------------------------- | ----- | ------------------------------- |
| `LANE_CHANGE_MIN_ACC_GAIN` | 0.1   | Same - will change lanes easily |
| `MERGE_ACC_GAIN`           | 0.8   | Higher - more willing to merge  |
| `MERGE_VEL_RATIO`          | 0.75  | Lower - merges at lower speeds  |
| `MERGE_TARGET_VEL`         | 30    | Higher target velocity          |

**Behavior**:

- Smaller following distances
- More frequent lane changes
- Higher target speeds
- Less consideration for other vehicles

---

## DefensiveVehicle Parameters

Inherits from `LinearVehicle` with defensive parameter tuning:

| Parameter                  | Value | Comparison to IDM                      |
| -------------------------- | ----- | -------------------------------------- |
| `LANE_CHANGE_MIN_ACC_GAIN` | 1.0   | Higher - needs big advantage to change |
| `MERGE_ACC_GAIN`           | 1.2   | Higher - more careful merging          |
| `MERGE_VEL_RATIO`          | 0.75  | Same                                   |
| `MERGE_TARGET_VEL`         | 30    | Same                                   |

**Behavior**:

- Larger following distances
- Fewer lane changes
- More predictable movements
- Better collision avoidance

---

## LinearVehicle Parameters

A simplified vehicle model with linear controllers:

| Parameter                 | Default                             | Description                    |
| ------------------------- | ----------------------------------- | ------------------------------ |
| `TIME_WANTED`             | 2.5                                 | Larger time gap than IDM (1.5) |
| `ACCELERATION_PARAMETERS` | [0.3, 0.3, 2.0]                     | Linear acceleration weights    |
| `STEERING_PARAMETERS`     | [KP_HEADING, KP_HEADING*KP_LATERAL] | Steering control gains         |

---

## Difficulty Tuning Guide

### 🟢 Easy Mode

```python
config = {
    "lanes_count": 4,
    "vehicles_count": 15,
    "vehicles_density": 0.5,
    "duration": 60,
    "other_vehicles_type": "highway_env.vehicle.behavior.DefensiveVehicle",
}
```

### 🟡 Medium Mode (Default)

```python
config = {
    "lanes_count": 4,
    "vehicles_count": 50,
    "vehicles_density": 1.0,
    "duration": 40,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
}
```

### 🔴 Hard Mode

```python
config = {
    "lanes_count": 3,
    "vehicles_count": 80,
    "vehicles_density": 2.0,
    "duration": 60,
    "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",
}
```

### 💀 Nightmare Mode

```python
config = {
    "lanes_count": 2,
    "vehicles_count": 100,
    "vehicles_density": 3.0,
    "duration": 120,
    "simulation_frequency": 30,
    "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",
}
```

---

## Custom Vehicle Classes

You can create custom vehicle behaviors by subclassing `IDMVehicle`:

```python
from highway_env.vehicle.behavior import IDMVehicle

class CustomAggressiveVehicle(IDMVehicle):
    """Very aggressive drivers with minimal safety margins"""

    # Longitudinal - faster acceleration, harder braking
    ACC_MAX = 10.0              # Max acceleration (default: 6.0)
    COMFORT_ACC_MAX = 5.0       # Comfortable accel (default: 3.0)
    COMFORT_ACC_MIN = -8.0      # Comfortable decel (default: -5.0)

    # Following distance - much closer
    DISTANCE_WANTED = 3.0       # Jam distance (default: ~10)
    TIME_WANTED = 0.8           # Time gap (default: 1.5)

    # Speed behavior
    DELTA = 3.0                 # Lower = more aggressive accel (default: 4.0)

    # Lane changing - very aggressive
    POLITENESS = 0.0            # No consideration for others
    LANE_CHANGE_MIN_ACC_GAIN = 0.05  # Tiny advantage triggers change
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 5.0  # Will cut people off hard
    LANE_CHANGE_DELAY = 0.5     # Quick decisions

class CustomDefensiveVehicle(IDMVehicle):
    """Very cautious drivers with large safety margins"""

    # Longitudinal - gentler
    ACC_MAX = 4.0
    COMFORT_ACC_MAX = 2.0
    COMFORT_ACC_MIN = -3.0

    # Following distance - much larger
    DISTANCE_WANTED = 15.0
    TIME_WANTED = 2.5

    # Speed behavior
    DELTA = 5.0                 # Higher = less aggressive

    # Lane changing - very conservative
    POLITENESS = 0.8            # Considers others
    LANE_CHANGE_MIN_ACC_GAIN = 0.5   # Needs big advantage
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 1.0  # Won't cut anyone off
    LANE_CHANGE_DELAY = 2.0     # Slow decisions
```

### Using Custom Classes

```python
# In your training script
from your_module import CustomAggressiveVehicle

env = gymnasium.make(
    "highway-v0",
    config={
        "other_vehicles_type": "your_module.CustomAggressiveVehicle",
    }
)
```

---

## Parameter Effect Summary

| Goal                           | Parameters to Adjust                                    |
| ------------------------------ | ------------------------------------------------------- |
| **More traffic**               | ↑ `vehicles_count`, ↑ `vehicles_density`                |
| **Less room to maneuver**      | ↓ `lanes_count`                                         |
| **Faster/slower traffic**      | Modify vehicle's `target_speed`                         |
| **More unpredictable traffic** | Use `AggressiveVehicle`, ↓ `POLITENESS`                 |
| **Harder lane changes**        | ↑ `vehicles_density`, ↓ `TIME_WANTED`                   |
| **More collisions**            | ↑ `vehicles_count`, ↓ `LANE_CHANGE_MAX_BRAKING_IMPOSED` |
| **Longer episodes**            | ↑ `duration`                                            |
| **More precise physics**       | ↑ `simulation_frequency`                                |

---

## Quick Reference Card

```python
# Maximum difficulty configuration
env = gymnasium.make(
    "highway-v0",
    config={
        # Road
        "lanes_count": 2,           # Fewer escape routes

        # Traffic
        "vehicles_count": 100,      # Maximum traffic
        "vehicles_density": 3.0,    # Very dense

        # Time
        "duration": 120,            # Longer survival needed
        "simulation_frequency": 30, # More precise (harder)

        # Behavior
        "other_vehicles_type": "highway_env.vehicle.behavior.AggressiveVehicle",
    },
    render_mode='rgb_array'
)
```

---

_Last updated: January 2026_
