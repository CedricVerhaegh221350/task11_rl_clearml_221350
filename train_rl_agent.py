"""
Train PPO on OT2GymEnv with strong logging + Weights & Biases.

Run example:
  python train_rl_agent.py --total-timesteps 2000000 --device cuda

Artifacts:
  runs/<run_id>/
    models/best_model.zip
    models/final_model.zip
    models/vecnormalize.pkl
    config.json
    tb/
"""

from __future__ import annotations

import os
import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import wandb

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv

import wandb
from wandb.integration.sb3 import WandbCallback
import atexit

from ot2_gym_wrapper import OT2GymEnv


@dataclass(frozen=True)
class TrainConfig:
    # Env
    max_steps: int = 400
    success_threshold_m: float = 0.03  # train for minimum criteria first
    vel_xy_max: float = 0.25
    vel_z_max: float = 0.15
    render: bool = False

    # PPO
    total_timesteps: int = 500_000
    learning_rate: float = 3e-4
    n_steps: int = 256
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0  # Entropy can prevent precision, especially early on.

    # Vec
    normalize_obs: bool = False
    normalize_reward: bool = False
    clip_obs: float = 10.0

    # System
    seed: int = 42
    n_envs: int = 2
    device: str = "cpu"

    # Eval
    eval_freq: int = 50_000
    n_eval_episodes: int = 30

    # Goal curriculum (optional) / Goal Box
    goal_curriculum_half_extent: tuple[float, float, float] | None = (0.03, 0.03, 0.02)

    # Run 1: Easy (Learning basic reaching)
    # success threshold: 0.03
    # goal box half extent: (0.03, 0.03, 0.02) meters
    # timesteps: 300_000 (this is the first one that might show real learning)
    # n_envs: 8, n_steps: 256

    # Run 2: Medium
    # success threshold: 0.02
    # goal box half extent: (0.06, 0.06, 0.04)
    # timesteps: 300_000

    # Run 3: Target
    # success threshold: 0.01
    # full envelope (no restriction)
    # timesteps: 500_000+



def make_env(cfg: TrainConfig, seed: int):
    def _init():
        return OT2GymEnv(
            max_steps=cfg.max_steps,
            success_threshold_m=cfg.success_threshold_m,
            seed=seed,
            render=cfg.render,
            vel_xy_max=cfg.vel_xy_max,
            vel_z_max=cfg.vel_z_max,
            enable_stuck_termination=False,
            goal_curriculum_half_extent=cfg.goal_curriculum_half_extent,
        )
    return _init


class MetricsCallback(BaseCallback):
    """
    Logs distance/success stats from infos to TensorBoard/W&B through SB3 logger.
    Works with VecEnv by reading infos list.
    """
    def __init__(self, log_every: int = 2000):
        super().__init__()
        self.log_every = int(log_every)
        self._dists = []
        self._succ_10mm = 0
        self._succ_5mm = 0
        self._succ_1mm = 0
        self._episodes = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        # record final episode dist on done
        for info, done in zip(infos, dones):
            if done and isinstance(info, dict) and "distance_to_goal" in info:
                d = float(info["distance_to_goal"])
                self._dists.append(d)
                self._episodes += 1
                if d < 0.01:
                    self._succ_10mm += 1
                if d < 0.005:
                    self._succ_5mm += 1
                if d < 0.001:
                    self._succ_1mm += 1

        if self.num_timesteps % self.log_every == 0 and self._episodes > 0:
            dists = np.array(self._dists, dtype=np.float64)
            self.logger.record("rollout/episodes", self._episodes)
            self.logger.record("rollout/final_error_mean_m", float(dists.mean()))
            self.logger.record("rollout/final_error_median_m", float(np.median(dists)))
            self.logger.record("rollout/success_rate_10mm", self._succ_10mm / self._episodes)
            self.logger.record("rollout/success_rate_5mm", self._succ_5mm / self._episodes)
            self.logger.record("rollout/success_rate_1mm", self._succ_1mm / self._episodes)

        return True


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=TrainConfig.total_timesteps)
    p.add_argument("--eval-freq", type=int, default=TrainConfig.eval_freq)
    p.add_argument("--n-eval-episodes", type=int, default=TrainConfig.n_eval_episodes)
    p.add_argument("--device", type=str, default=TrainConfig.device)
    p.add_argument("--n-envs", type=int, default=TrainConfig.n_envs)
    p.add_argument("--success-threshold", type=float, default=TrainConfig.success_threshold_m)
    args = p.parse_args()

    return TrainConfig(
        total_timesteps=args.total_timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        device=args.device,
        n_envs=args.n_envs,
        success_threshold_m=args.success_threshold,
    )


def main():
    cfg = parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("runs") / f"ppo_ot2_{run_id}"
    (out_dir / "models").mkdir(parents=True, exist_ok=True)
    (out_dir / "tb").mkdir(parents=True, exist_ok=True)

    # --- W&B init ---
    wandb_run = wandb.init(
        project="datalab-task11-ot2-rl",
        entity="221350-buas",
        name=f"ppo_ot2_{run_id}",
        notes="PPO training on OT2GymEnv",
        config=asdict(cfg),
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )

    # --- Ensure wandb finishes on exit ---
    atexit.register(lambda: wandb.finish())


    # Save config.json so eval can match training exactly
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # --- training env ---
    train_env = DummyVecEnv([make_env(cfg, cfg.seed + i) for i in range(cfg.n_envs)])
    train_env = VecNormalize(
        train_env,
        norm_obs=cfg.normalize_obs,
        norm_reward=cfg.normalize_reward,
        clip_obs=cfg.clip_obs,
    )

    # --- eval env ---
    eval_env = DummyVecEnv([make_env(cfg, cfg.seed + 10_000)])
    eval_env = VecNormalize(
        eval_env,
        norm_obs=cfg.normalize_obs,
        norm_reward=False,
        clip_obs=cfg.clip_obs,
    )
    eval_env.training = False
    # Only sync obs_rms if obs normalization is enabled
    if cfg.normalize_obs:
        eval_env.obs_rms = train_env.obs_rms

    # --- Callbacks ---
    # --- Eval Callback ---
    eval_callback = EvalCallback(
        eval_env,                                       # Evaluation environment
        best_model_save_path=str(out_dir / "models"),   # Where to save best model
        log_path=str(out_dir / "eval_logs"),            # Where to save eval logs
        eval_freq=cfg.eval_freq,                        # Evaluate every N steps
        n_eval_episodes=cfg.n_eval_episodes,            # Number of episodes per eval
        deterministic=True,                             # Use deterministic actions 
        render=False,                                   # No render during eval
    )

    # --- Custom Metrics Callback ---
    metrics_callback = MetricsCallback(log_every=5000) # Log custom metrics every N steps

    # --- W&B Callback ---
    wandb_callback = WandbCallback(
        model_save_freq=cfg.eval_freq,              # Save model every N steps
        model_save_path=str(out_dir / "models"),    # Where to save models
        verbose=2,                                  # Verbosity level
    )

    # --- Combine Callbacks ---
    callback = CallbackList([eval_callback, metrics_callback, wandb_callback])


    # --- Model ---
    model = PPO(
        "MlpPolicy",                                # Policy architecture
        env=train_env,                              # Training environment
        device=cfg.device,                          # "auto", "cpu", or "cuda"
        learning_rate=cfg.learning_rate,            # Adam learning rate
        n_steps=cfg.n_steps,                        # Steps per rollout
        batch_size=cfg.batch_size,                  # Minibatch size
        n_epochs=cfg.n_epochs,                      # Number of epochs
        gamma=cfg.gamma,                            # Discount factor
        gae_lambda=cfg.gae_lambda,                  # GAE lambda
        clip_range=cfg.clip_range,                  # Clipping range
        ent_coef=cfg.ent_coef,                      # Entropy coefficient
        verbose=1,                                  # Verbosity level
        tensorboard_log=str(out_dir / "tb"),        # TensorBoard log dir
        seed=cfg.seed,                              # RNG seed  
    )


    # --- Train ---
    print(f"[INFO] Training start: out_dir={out_dir}")
    model.learn(total_timesteps=cfg.total_timesteps, callback=callback, progress_bar=True)

    # Save final model + vecnorm
    final_model_path = out_dir / "models" / "final_model.zip"
    vecnorm_path = out_dir / "models" / "vecnormalize.pkl"

    model.save(str(final_model_path))
    train_env.save(str(vecnorm_path))

    # Upload to W&B as files
    wandb.save(str(final_model_path))
    wandb.save(str(vecnorm_path))

    wandb_run.finish()

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
