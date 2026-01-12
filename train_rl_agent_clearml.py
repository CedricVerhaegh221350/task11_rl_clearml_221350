from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

from clearml import Task


def _find_latest_run_dir(runs_root: Path) -> Optional[Path]:
    if not runs_root.exists():
        return None
    candidates = [p for p in runs_root.iterdir() if p.is_dir() and p.name.startswith("ppo_ot2_")]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main():
    # 1) Create ClearML task
    task = Task.init(
        project_name="DataLab-Task11-RL/CedricVerhaegh_221350",
        task_name="PPO_OT2_Train",
    )

    # 2) Expose WANDB key as a ClearML hyperparameter (edit in UI)
    params = {
        "WANDB": {
            "api_key": "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",   # set this in ClearML UI (Configuration -> Hyperparameters)
        }
    }
    params = task.connect(params)

    # 3) Set docker image
    task.set_base_docker("deanis/2023y2b-rl:latest")

    # 4) Enqueue remote
    task.execute_remotely(queue_name="default")

    # --- Everything below runs on the remote worker ---

    # 5) Apply W&B key BEFORE importing code that uses wandb
    wandb_key = (params.get("WANDB", {}) or {}).get("api_key", "")
    if wandb_key:
        os.environ["WANDB_API_KEY"] = wandb_key
        print("[ClearML] WANDB_API_KEY set from ClearML Hyperparameters.")
    else:
        print("[ClearML][WARN] WANDB api_key is empty. W&B may fail to login remotely.")

    # 6) Run your existing training script
    import train_rl_agent  # must be in repo root / same folder
    train_rl_agent.main()

    # 7) Upload latest run folder
    runs_root = Path("runs")
    latest = _find_latest_run_dir(runs_root)
    if latest is not None:
        zip_path = shutil.make_archive(str(latest), "zip", root_dir=str(latest))
        task.upload_artifact(name="runs_folder_zip", artifact_object=zip_path)
        print(f"[ClearML] Uploaded artifact: {zip_path}")
    else:
        print("[ClearML] No runs/ppo_ot2_* folder found to upload.")


if __name__ == "__main__":
    main()
