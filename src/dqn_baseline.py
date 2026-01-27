import gymnasium
import highway_env
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm
from highway_env.vehicle.behavior import IDMVehicle

class DMVehicle(IDMVehicle):
    ACC_MAX = 8.0
    ACC_MIN = -3.5
    tau = 0.8
    delta = 3.0
    POLITENESS = 1
    LANE_CHANGE_MIN_ACC_GAIN = 0.1
    LANE_CHANGE_MAX_BRAKING_IMPOSED = 3.0

env = gymnasium.make(
  "highway-v0",
  config={
    "lanes_count": 4, 
    "vehicles_count": 25,  
    "vehicles_density": 1,
    "duration": 60,
    "simulation_frequency":30,
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
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

model = DQN('MlpPolicy', env,
              policy_kwargs=dict(net_arch=[256, 256]),
              learning_rate=5e-4,
              buffer_size=15000,
              learning_starts=200,
              batch_size=32,
              gamma=0.8,
              train_freq=1,
              gradient_steps=1,
              target_update_interval=50,
              verbose=1,
              tensorboard_log="highway_dqn/",
              device=device)

'''
model.learn(int(100000), callback=TqdmCallback())
model.save("highway_dqn/model")
'''
# Load and test saved model
model = DQN.load("highway_dqn/model", device=device)
while True:
  done = truncated = False
  obs, info = env.reset()
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    env.render()