"""
Minimal local evaluation for a trained PPO policy on OT2GymEnv.

New behavior:
- You can point it to a run folder: --run runs/ppo_ot2_YYYYMMDD_HHMMSS
- It will auto-load:
    - config.json (env params)
    - models/vecnormalize.pkl
    - models/best_model.zip (preferred) or models/final_model.zip
- CLI flags can still override config values.

Example:
  python test_rl_agent_local.py --run runs/ppo_ot2_20260112_153000 --episodes 50 --render

Or explicit paths:
  python test_rl_agent_local.py --model ... --vecnorm ... --episodes 30
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from ot2_gym_wrapper import OT2GymEnv


# -------------------------
# Client scoring table
# -------------------------
def client_score_from_error(error_m: float) -> int:
    if error_m < 0.001:
        return 8
    if error_m < 0.005:
        return 6
    if error_m < 0.01:
        return 4
    return 0


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class EvalConfig:
    run_dir: Optional[Path]
    model_path: Path
    vecnorm_path: Path

    episodes: int = 30
    max_steps: int = 400
    success_threshold_m: float = 0.001  # evaluation threshold (default 1 mm)
    seed: int = 123
    deterministic: bool = True
    render: bool = False

    # Must match training (unless deliberately testing mismatch)
    vel_xy_max: float = 0.25
    vel_z_max: float = 0.15

    device: str = "cpu"


def _load_train_config_json(run_dir: Path) -> Dict:
    cfg_path = run_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _auto_paths_from_run(run_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Return (model_path, vecnorm_path) if found, else (None, None).
    Prefers best_model.zip; falls back to final_model.zip.
    """
    models_dir = run_dir / "models"
    vecnorm = models_dir / "vecnormalize.pkl"
    if not vecnorm.exists():
        vecnorm = None

    best = models_dir / "best_model.zip"
    final = models_dir / "final_model.zip"
    model = best if best.exists() else (final if final.exists() else None)

    return model, vecnorm


def make_eval_env(cfg: EvalConfig) -> VecNormalize:
    vec_env = make_vec_env(
        env_id=lambda: OT2GymEnv(
            max_steps=cfg.max_steps,
            success_threshold_m=cfg.success_threshold_m,
            seed=cfg.seed,
            render=cfg.render,
            vel_xy_max=cfg.vel_xy_max,
            vel_z_max=cfg.vel_z_max,
            enable_stuck_termination=False,
        ),
        n_envs=1,
        seed=cfg.seed,
    )

    env = VecNormalize.load(str(cfg.vecnorm_path), vec_env)
    env.training = False
    env.norm_reward = False
    return env


def run_eval(cfg: EvalConfig) -> Dict[str, float]:
    env = make_eval_env(cfg)
    model = PPO.load(str(cfg.model_path), env=env, device=cfg.device)

    final_errors: List[float] = []
    client_scores: List[int] = []
    successes = 0

    for ep in range(cfg.episodes):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < cfg.max_steps:
            action, _ = model.predict(obs, deterministic=cfg.deterministic)
            obs, rewards, dones, infos = env.step(action)
            done = bool(dones[0])
            steps += 1

        ep_info = infos[0]
        final_error = float(ep_info["distance_to_goal"])
        final_errors.append(final_error)

        score = client_score_from_error(final_error)
        client_scores.append(score)

        if final_error < cfg.success_threshold_m:
            successes += 1

        term_success = bool(ep_info.get("terminated_success", False))
        term_stuck = bool(ep_info.get("terminated_stuck", False))

        print(
            f"Episode {ep+1:03d}/{cfg.episodes} | steps={steps:3d} | "
            f"final_error={final_error:.6f} m | score={score} | "
            f"success={term_success} stuck={term_stuck}"
        )

    env.close()

    final_errors_np = np.array(final_errors, dtype=np.float64)
    client_scores_np = np.array(client_scores, dtype=np.int32)

    results = {
        "episodes": float(cfg.episodes),
        "success_rate": float(successes / max(cfg.episodes, 1)),
        "mean_final_error_m": float(np.mean(final_errors_np)),
        "median_final_error_m": float(np.median(final_errors_np)),
        "max_final_error_m": float(np.max(final_errors_np)),
        "mean_client_score": float(np.mean(client_scores_np)),
    }

    print("\n=== Evaluation Summary ===")
    print(f"Run dir:              {cfg.run_dir if cfg.run_dir else '(none)'}")
    print(f"Model:               {cfg.model_path}")
    print(f"VecNormalize:         {cfg.vecnorm_path}")
    print(f"Episodes:             {int(results['episodes'])}")
    print(f"Success rate:         {results['success_rate']*100:.1f}%  (error < {cfg.success_threshold_m} m)")
    print(f"Mean final error:     {results['mean_final_error_m']:.6f} m")
    print(f"Median final error:   {results['median_final_error_m']:.6f} m")
    print(f"Max final error:      {results['max_final_error_m']:.6f} m")
    print(f"Mean client score:    {results['mean_client_score']:.2f} / 8")

    brackets = {
        "<1mm (8pts)": np.sum(final_errors_np < 0.001),
        "1-5mm (6pts)": np.sum((final_errors_np >= 0.001) & (final_errors_np < 0.005)),
        "5-10mm (4pts)": np.sum((final_errors_np >= 0.005) & (final_errors_np < 0.01)),
        ">=10mm (0pts)": np.sum(final_errors_np >= 0.01),
    }
    print("\nError bracket counts:")
    for k, v in brackets.items():
        print(f"  {k:13s}: {int(v)}")

    return results


def parse_args() -> EvalConfig:
    p = argparse.ArgumentParser(description="Evaluate a trained PPO agent on OT2GymEnv.")
    p.add_argument("--run", type=str, default=None, help="Run folder (contains config.json and models/).")
    p.add_argument("--model", type=str, default=None, help="Path to SB3 model zip.")
    p.add_argument("--vecnorm", type=str, default=None, help="Path to VecNormalize stats (vecnormalize.pkl).")

    p.add_argument("--episodes", type=int, default=EvalConfig.episodes)
    p.add_argument("--max-steps", type=int, default=EvalConfig.max_steps)
    p.add_argument("--threshold", type=float, default=EvalConfig.success_threshold_m)
    p.add_argument("--seed", type=int, default=EvalConfig.seed)
    p.add_argument("--deterministic", action="store_true", default=EvalConfig.deterministic)
    p.add_argument("--render", action="store_true", default=EvalConfig.render)

    p.add_argument("--vel-xy-max", type=float, default=EvalConfig.vel_xy_max)
    p.add_argument("--vel-z-max", type=float, default=EvalConfig.vel_z_max)

    p.add_argument("--device", type=str, default=EvalConfig.device)

    args = p.parse_args()

    run_dir = Path(args.run) if args.run else None
    train_cfg = {}

    auto_model = None
    auto_vecnorm = None
    if run_dir:
        if not run_dir.exists():
            raise FileNotFoundError(f"Run dir not found: {run_dir}")
        train_cfg = _load_train_config_json(run_dir)
        auto_model, auto_vecnorm = _auto_paths_from_run(run_dir)

    model_path = Path(args.model) if args.model else auto_model
    vecnorm_path = Path(args.vecnorm) if args.vecnorm else auto_vecnorm
    if model_path is None:
        raise FileNotFoundError("Model path not provided and could not auto-detect from --run.")
    if vecnorm_path is None:
        raise FileNotFoundError("VecNormalize path not provided and could not auto-detect from --run.")

    # Defaults from training config.json, then overridden by CLI if provided
    max_steps = int(args.max_steps) if args.max_steps is not None else int(train_cfg.get("max_steps", 400))
    vel_xy_max = float(args.vel_xy_max) if args.vel_xy_max is not None else float(train_cfg.get("vel_xy_max", 0.20))
    vel_z_max = float(args.vel_z_max) if args.vel_z_max is not None else float(train_cfg.get("vel_z_max", 0.10))

    return EvalConfig(
        run_dir=run_dir,
        model_path=Path(model_path),
        vecnorm_path=Path(vecnorm_path),
        episodes=int(args.episodes),
        max_steps=max_steps,
        success_threshold_m=float(args.threshold),
        seed=int(args.seed),
        deterministic=bool(args.deterministic),
        render=bool(args.render),
        vel_xy_max=vel_xy_max,
        vel_z_max=vel_z_max,
        device=str(args.device),
    )


def main() -> None:
    cfg = parse_args()

    if not cfg.model_path.exists():
        raise FileNotFoundError(f"Model not found: {cfg.model_path}")
    if not cfg.vecnorm_path.exists():
        raise FileNotFoundError(f"VecNormalize stats not found: {cfg.vecnorm_path}")

    run_eval(cfg)


if __name__ == "__main__":
    main()
