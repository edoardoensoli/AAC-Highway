"""
Quick test: 20k step per verificare che PPO esplori e impari a schivare.
Stampa distribuzione azioni + entropia + headway reward ogni 2k step.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from ppo_accel.env import HighwayEnvWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np, torch

ACTIONS = {0: "←", 1: "=", 2: "→", 3: "▲", 4: "▼"}
TOTAL = 20_000
CHECK_EVERY = 2_000


class ProgressCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self._t0 = None
        self._last_check = 0
        self._ep_returns = []
        self._headway_penalties = []

    def _on_training_start(self):
        self._t0 = time.time()

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "headway_reward" in info:
                self._headway_penalties.append(info["headway_reward"])
            if "episode" in info:
                self._ep_returns.append(info["episode"]["r"])

        step = self.num_timesteps
        if step - self._last_check >= CHECK_EVERY:
            self._last_check = step
            self._print_status(step)
        return True

    def _print_status(self, step):
        elapsed = time.time() - self._t0
        sps = step / elapsed if elapsed > 0 else 0
        pct = step / TOTAL * 100
        eta = (TOTAL - step) / sps if sps > 0 else 0

        # Distribuzione azioni (300 step stocastici)
        env = self.model.get_env()
        obs = env.reset()
        counts = {a: 0 for a in range(5)}
        for _ in range(300):
            action, _ = self.model.predict(obs, deterministic=False)
            counts[int(action[0])] += 1
            obs, _, d, _ = env.step(action)
            if d[0]:
                obs = env.reset()

        # Entropia
        obs_t = torch.as_tensor(env.reset()).float()
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_t)
            ent = dist.entropy().item()
        ent_pct = ent / np.log(5) * 100

        # Reward e headway recenti
        recent = self._ep_returns[-20:] if self._ep_returns else [0]
        mean_r = np.mean(recent)
        hw = self._headway_penalties[-200:] if self._headway_penalties else [0]
        mean_hw = np.mean(hw)

        bar_w = 20
        filled = int(bar_w * step / TOTAL)
        bar = "█" * filled + "░" * (bar_w - filled)

        print(f"  {bar} {pct:>5.1f}%  step {step:>6,}/{TOTAL:,}  "
              f"{sps:.0f} sps  ETA {eta:.0f}s")
        print(f"    Azioni:  ", end="")
        for a in range(5):
            print(f"{ACTIONS[a]} {counts[a]/3:.0f}%  ", end="")
        print(f"  Ent: {ent_pct:.0f}%")
        print(f"    Reward: {mean_r:+.2f}  Headway penalty: {mean_hw:.3f}")
        if ent_pct < 30:
            print(f"    ⚠️  POLICY COLLAPSE!")
        print()


if __name__ == "__main__":
    env = DummyVecEnv([lambda: Monitor(HighwayEnvWrapper())])

    print(f"\n  Quick test PPO - {TOTAL:,} step")
    print(f"  Obs: {env.observation_space.shape}  Actions: 5")
    cfg = HighwayEnvWrapper.FIXED_CONFIG
    print(f"  Reward: crash={cfg['collision_reward']}"
          f"  speed={cfg['high_speed_reward']}"
          f"  headway_weight={HighwayEnvWrapper.HEADWAY_REWARD_WEIGHT}")
    print()

    model = PPO(
        "MlpPolicy", env, verbose=0,
        ent_coef=0.05, n_epochs=4, n_steps=512, gamma=0.95,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
    )

    cb = ProgressCallback()
    model.learn(total_timesteps=TOTAL, callback=cb)

    env.close()
    print("  ✓ Test completato!")
