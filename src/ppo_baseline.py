import gymnasium
import highway_env
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

env = gymnasium.make(
  "highway-v0",
  config={
    "lanes_count": 3, 
    "vehicles_count": 25,  
    "vehicles_density": 1,
    "duration": 60,  
    },
  render_mode='rgb_array'
)

# Device detection
if torch.cuda.is_available():
    device = "cuda"
    print("Using CUDA GPU")
else:
    device = "cpu"
    print("Using CPU")

class TqdmCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.pbar = None
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.locals['total_timesteps'])
    
    def _on_step(self):
        self.pbar.update(1)
        return True
    
    def _on_training_end(self):
        self.pbar.close()

model = PPO('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              n_steps=2048,
              batch_size=64,
              gamma=0.8,
              gae_lambda=0.95,
              clip_range=0.2,
              ent_coef=0.01,
              verbose=1,
              tensorboard_log="highway_ppo/",
              device=device)


model.learn(int(50000), callback=TqdmCallback())
model.save("highway_ppo/model")

# Load and test saved model
model = PPO.load("highway_ppo/model", device=device)
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()