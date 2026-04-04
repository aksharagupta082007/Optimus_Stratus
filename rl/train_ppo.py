import os
import sys

# Ensure the project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from env.cubesat_env import CubeSatEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from dataclasses import dataclass

@dataclass
class PPORLConfig:
    # Force the environment to compress telemetries into a 1D Numpy Array instead of a Dictionary!
    observation_type: str = "vector"
    include_action_mask_in_obs: bool = True

def train():
    print("Initializing CubeSat Environment...")
    env = CubeSatEnv(rl_config=PPORLConfig())
    
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    print("Configuring PPO Agent with Micro-MLP Architecture [64, 64]...")
    
    # Custom MLP architecture to keep the model < 1MB for the Raspberry Pi
    policy_kwargs = dict(net_arch=[64, 64])

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=logs_dir,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        gamma=0.99
    )

    # Save a checkpoint every 10,000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, 
        save_path=models_dir,
        name_prefix="ppo_cubesat_ckpt"
    )

    total_timesteps = 50000
    print(f"Starting Training Session for {total_timesteps} steps...")
    
    model.learn(
        total_timesteps=total_timesteps, 
        callback=checkpoint_callback,
        tb_log_name="PPO_Micro"
    )

    final_model_path = os.path.join(models_dir, "cubesat_ppo.zip")
    model.save(final_model_path)
    print(f"Training Complete! Final model saved to: {final_model_path}")

if __name__ == "__main__":
    train()
