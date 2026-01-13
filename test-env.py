import gymnasium as gym
import highway_env  # Necessario per registrare gli ambienti

env_type = ['merge-v0', 'roundabout-v0', 'highway-v0', 'intersection-v0', 'parking-v0']

env = gym.make(env_type[4], render_mode='human')

obs, info = env.reset()
done = truncated = False

while not (done or truncated):
    action = env.action_space.sample()  # Azione casuale
    obs, reward, done, truncated, info = env.step(action)

env.close()