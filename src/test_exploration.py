"""Quick test: verify policy doesn't collapse with new hyperparams."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from ppo_accel.env import HighwayEnvWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

ACTION_NAMES = {0: "L_LEFT", 1: "IDLE", 2: "L_RIGHT", 3: "FASTER", 4: "SLOWER"}

def test_model(model, env, label, n=500):
    obs = env.reset()
    counts = {a: 0 for a in range(5)}
    for _ in range(n):
        action, _ = model.predict(obs, deterministic=False)
        counts[int(action[0])] += 1
        obs, r, d, i = env.step(action)
        if d[0]: obs = env.reset()
    
    obs_t = torch.as_tensor(env.reset()).float().to(model.device)
    dist = model.policy.get_distribution(obs_t)
    ent = dist.entropy().item()
    
    print(f"\n  {label}")
    for a in range(5):
        c = counts[a]
        bar = "#" * int(c / n * 40)
        print(f"    {ACTION_NAMES[a]:>8}: {c:>3} ({c/n*100:>5.1f}%) {bar}")
    print(f"    Entropia: {ent:.4f} / {np.log(5):.4f} = {ent/np.log(5)*100:.0f}%")
    dominant = max(counts, key=counts.get)
    if counts[dominant] / n > 0.6:
        print(f"    !! COLLAPSE su {ACTION_NAMES[dominant]}")
    else:
        print(f"    OK - policy distribuita")
    return ent

env = DummyVecEnv([lambda: Monitor(HighwayEnvWrapper())])

print("=" * 60)
print("  Test esplorazione: ent_coef=0.05, n_epochs=4")
print("=" * 60)

model = PPO("MlpPolicy", env, verbose=0, ent_coef=0.05, n_epochs=4, n_steps=512,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))
test_model(model, env, "FRESCO (0 step)")

model.learn(total_timesteps=5000)
test_model(model, env, "DOPO 5k step")

model.learn(total_timesteps=15000)
test_model(model, env, "DOPO 20k step")

model.learn(total_timesteps=30000)
test_model(model, env, "DOPO 50k step")

print("\n" + "=" * 60)
print("  Confronto: ent_coef=0.01, n_epochs=10 (VECCHIO)")
print("=" * 60)

model_old = PPO("MlpPolicy", env, verbose=0, ent_coef=0.01, n_epochs=10, n_steps=256,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))
model_old.learn(total_timesteps=50000)
test_model(model_old, env, "VECCHIO dopo 50k step")

env.close()
print("\nDone!")
