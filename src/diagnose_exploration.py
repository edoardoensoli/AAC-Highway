"""
Diagnostic: why doesn't PPO explore?
Compare action distributions of fresh vs trained models.
"""
import sys, os, numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from ppo_accel.env import HighwayEnvWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO

ACTION_NAMES = {0: "LANE_LEFT", 1: "IDLE", 2: "LANE_RIGHT", 3: "FASTER", 4: "SLOWER"}
NUM_TEST_STEPS = 500


def analyze_model(model, env, label, num_steps=NUM_TEST_STEPS):
    """Run model for N steps and analyze action distribution + policy entropy."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    actions_count = {a: 0 for a in range(5)}
    all_entropies = []
    all_probs = []
    obs = env.reset()

    for step in range(num_steps):
        # Get action probabilities
        obs_tensor = torch.as_tensor(obs).float().to(model.device)
        dist = model.policy.get_distribution(obs_tensor)
        probs = dist.distribution.probs.detach().cpu().numpy()[0]
        entropy = dist.entropy().detach().cpu().item()
        all_probs.append(probs)
        all_entropies.append(entropy)

        # Take stochastic action
        action, _ = model.predict(obs, deterministic=False)
        obs, r, d, info = env.step(action)
        actions_count[int(action[0])] += 1
        if d[0]:
            obs = env.reset()

    # Also check deterministic actions
    actions_det = {a: 0 for a in range(5)}
    obs = env.reset()
    for step in range(num_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, d, info = env.step(action)
        actions_det[int(action[0])] += 1
        if d[0]:
            obs = env.reset()

    print(f"\n  Azioni STOCASTICHE ({num_steps} step):")
    for a in range(5):
        c = actions_count[a]
        bar = "#" * int(c / num_steps * 50)
        print(f"    {ACTION_NAMES[a]:>12}: {c:>4} ({c/num_steps*100:>5.1f}%) {bar}")

    print(f"\n  Azioni DETERMINISTICHE ({num_steps} step):")
    for a in range(5):
        c = actions_det[a]
        bar = "#" * int(c / num_steps * 50)
        print(f"    {ACTION_NAMES[a]:>12}: {c:>4} ({c/num_steps*100:>5.1f}%) {bar}")

    mean_probs = np.mean(all_probs, axis=0)
    print(f"\n  Probabilita medie della policy:")
    for a in range(5):
        bar = "#" * int(mean_probs[a] * 50)
        print(f"    {ACTION_NAMES[a]:>12}: {mean_probs[a]:.4f} {bar}")

    print(f"\n  Entropia policy:")
    print(f"    Media:  {np.mean(all_entropies):.4f}")
    print(f"    Min:    {np.min(all_entropies):.4f}")
    print(f"    Max:    {np.max(all_entropies):.4f}")
    print(f"    Max teorico (5 azioni uniformi): {np.log(5):.4f}")
    print(f"    Rapporto: {np.mean(all_entropies)/np.log(5)*100:.1f}% del massimo")

    # Check: which action would a uniform random policy pick?
    print(f"\n  Confronto con random uniforme:")
    print(f"    Random: 20% per azione")
    dominant = max(actions_count, key=actions_count.get)
    print(f"    Policy: {ACTION_NAMES[dominant]} domina con {actions_count[dominant]/num_steps*100:.1f}%")

    if actions_count[dominant] / num_steps > 0.5:
        print(f"    ⚠️  POLICY COLLAPSE! L'agente usa quasi solo {ACTION_NAMES[dominant]}")
    
    return mean_probs, np.mean(all_entropies)


def main():
    env = DummyVecEnv([lambda: Monitor(HighwayEnvWrapper())])

    # 1. Modello fresco
    print("\n" + "🔬 DIAGNOSTICA ESPLORAZIONE PPO" + "\n")
    
    model_fresh = PPO("MlpPolicy", env, verbose=0, ent_coef=0.01,
                       policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))
    analyze_model(model_fresh, env, "MODELLO FRESCO (ent_coef=0.01)")

    # 2. Quick train: 2000 steps to see how fast collapse happens
    print("\n\n📊 Training veloce per vedere quanto velocemente collassa...")
    model_quick = PPO("MlpPolicy", env, verbose=0, ent_coef=0.01, n_steps=256,
                       policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))
    model_quick.learn(total_timesteps=2000)
    analyze_model(model_quick, env, "DOPO 2000 STEP (ent_coef=0.01)")

    model_quick.learn(total_timesteps=8000)
    analyze_model(model_quick, env, "DOPO 10000 STEP (ent_coef=0.01)")

    # 3. Trained model if exists
    import glob
    models_100k = glob.glob("ppo_accel_1M/checkpoint_step100000*")
    if models_100k:
        model_path = models_100k[0].replace(".zip", "")
        model_trained = PPO.load(model_path, env=env)
        analyze_model(model_trained, env, f"MODELLO ALLENATO 100k: {model_path}")

    # 4. Test con entropy alta
    print("\n\n🧪 TEST: ent_coef=0.1 (10x piu alto)")
    model_high_ent = PPO("MlpPolicy", env, verbose=0, ent_coef=0.1, n_steps=256,
                          policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])))
    model_high_ent.learn(total_timesteps=10000)
    analyze_model(model_high_ent, env, "DOPO 10000 STEP (ent_coef=0.1)")

    # 5. Reward analysis
    print(f"\n\n{'='*60}")
    print(f"  📐 ANALISI REWARD PER-STEP")
    print(f"{'='*60}")
    env2 = HighwayEnvWrapper()
    obs, _ = env2.reset()
    rewards_by_action = {a: [] for a in range(5)}
    for action in range(5):
        env2.reset()
        for _ in range(20):
            _, r, term, trunc, _ = env2.step(action)
            rewards_by_action[action].append(r)
            if term or trunc:
                break
    
    print(f"\n  Reward per azione (primi 20 step):")
    for a in range(5):
        rews = rewards_by_action[a]
        print(f"    {ACTION_NAMES[a]:>12}: mean={np.mean(rews):>6.2f} | "
              f"sum={np.sum(rews):>7.2f} | steps={len(rews)}")
    
    print(f"\n  ➡️  Se FASTER da reward alta ogni step,")
    print(f"     PPO converge subito su FASTER e non esplora mai altro.")
    print(f"     DQN invece usa epsilon-greedy (azione random 10-100% del tempo)")
    print(f"     e impara dal replay buffer.")

    env.close()
    env2.close()


if __name__ == "__main__":
    main()
