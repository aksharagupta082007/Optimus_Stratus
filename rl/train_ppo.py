"""
PPO Training Script for CubeSat Resource Management Agent.

Uses Stable-Baselines3 with a compact MLP architecture (64x64)
designed for eventual deployment on a Raspberry Pi (< 1MB model).

Usage:
    python -m rl.train_ppo                          # Default 200k steps
    python -m rl.train_ppo --timesteps 500000       # Custom step count
    python -m rl.train_ppo --resume                 # Resume from last checkpoint
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

# Ensure project root is importable
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# ─── Gymnasium wrapper ───────────────────────────────────────────────
import gymnasium as gym
from gymnasium import spaces

from env.cubesat_env import CubeSatEnv, LocalRLConfig, LocalMissionConfig


class CubeSatGymWrapper(gym.Env):
    """
    Thin Gymnasium-compliant wrapper around CubeSatEnv that guarantees:
      1. Observations are flat float32 vectors (required by SB3 MLP).
      2. Action masking info is passed through the `info` dict.
      3. Observations and rewards are numerically stable (no NaN/Inf).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self):
        super().__init__()

        # Force vector observations for flat MLP input
        rl_config = LocalRLConfig(observation_type="vector")
        mission_config = LocalMissionConfig(randomize_initial_phase=True)
        self.env = CubeSatEnv(rl_config=rl_config, mission_config=mission_config)

        # Probe the observation shape from a dummy reset
        obs, _ = self.env.reset()
        obs_array = self._to_array(obs)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=obs_array.shape,
            dtype=np.float32,
        )
        self.action_space = self.env.action_space

    def _to_array(self, obs) -> np.ndarray:
        """Safely convert any observation format to a flat float32 array."""
        if isinstance(obs, np.ndarray):
            arr = obs.astype(np.float32).flatten()
        elif isinstance(obs, (list, tuple)):
            arr = np.array(obs, dtype=np.float32).flatten()
        elif isinstance(obs, dict):
            # If it's a bundle dict, concatenate the vector key
            if "vector" in obs:
                arr = np.array(obs["vector"], dtype=np.float32).flatten()
            else:
                arr = np.array(list(obs.values()), dtype=np.float32).flatten()
        else:
            arr = np.array([float(obs)], dtype=np.float32)

        # Sanitize NaN / Inf
        arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        return arr

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._to_array(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        obs_arr = self._to_array(obs)

        # Clamp reward to prevent gradient explosions
        reward = float(np.clip(reward, -100.0, 100.0))

        return obs_arr, reward, terminated, truncated, info

    def render(self):
        return self.env.render()


# ─── Training logic ──────────────────────────────────────────────────

def make_env():
    """Factory function for SubprocVecEnv compatibility."""
    def _init():
        return CubeSatGymWrapper()
    return _init


def train(
    total_timesteps: int = 200_000,
    save_dir: str = "models",
    log_dir: str = "logs/ppo_cubesat",
    resume: bool = False,
):
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import (
        CheckpointCallback,
        EvalCallback,
    )
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Paths
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    model_file = save_path / "cubesat_ppo.zip"

    # ── Environment ──
    print("=" * 60)
    print("  CubeSat PPO Training Pipeline")
    print("=" * 60)
    print(f"  Total timesteps : {total_timesteps:,}")
    print(f"  Model save path : {model_file}")
    print(f"  TensorBoard logs: {log_path}")
    print("=" * 60)

    env = DummyVecEnv([make_env()])

    # Separate eval env for periodic evaluation
    eval_env = DummyVecEnv([make_env()])

    # ── Model ──
    if resume and model_file.exists():
        print(f"\n>> Resuming training from {model_file}")
        model = PPO.load(str(model_file), env=env, tensorboard_log=str(log_path))
    else:
        print("\n>> Initializing new PPO model...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            # ── Micro-MLP: 64x64 for Raspberry Pi deployment ──
            policy_kwargs=dict(
                net_arch=dict(pi=[64, 64], vf=[64, 64]),
            ),
            learning_rate=3e-4,
            n_steps=2048,          # Steps per rollout buffer collection
            batch_size=64,         # Mini-batch size for gradient updates
            n_epochs=10,           # PPO epochs per rollout
            gamma=0.99,            # Discount factor
            gae_lambda=0.95,       # GAE lambda
            clip_range=0.2,        # PPO clipping parameter
            ent_coef=0.05,         # INCREASED: Entropy bonus to prevent early deterministic collapse
            vf_coef=0.5,           # Value function loss weight
            max_grad_norm=0.5,     # Gradient clipping
            verbose=1,
            tensorboard_log=str(log_path),
            seed=42,
        )

    # ── Callbacks ──
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,
        save_path=str(save_path / "checkpoints"),
        name_prefix="cubesat_ppo",
        verbose=1,
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best"),
        log_path=str(log_path / "eval"),
        eval_freq=20_000,
        n_eval_episodes=5,
        deterministic=True,
        verbose=1,
    )

    # ── Train ──
    print("\n>> Training started...\n")
    t0 = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    elapsed = time.time() - t0
    print(f"\n>> Training complete in {elapsed / 60:.1f} minutes.")

    # ── Save final model ──
    model.save(str(model_file))
    print(f">> Final model saved to: {model_file}")

    # ── Quick sanity check ──
    print("\n>> Running quick inference test...")
    test_env = CubeSatGymWrapper()
    obs, info = test_env.reset()
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    print(f"   Test episode: {steps} steps, total reward = {total_reward:.2f}")
    print("\n>> Done! You can visualize training curves with:")
    print(f"   tensorboard --logdir {log_path}")

    return model


# ─── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for CubeSat")
    parser.add_argument(
        "--timesteps", type=int, default=200_000,
        help="Total training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from last saved model",
    )
    args = parser.parse_args()

    train(total_timesteps=args.timesteps, resume=args.resume)
