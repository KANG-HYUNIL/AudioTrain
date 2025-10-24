"""
Experiment orchestrator to run 6 scenarios and log via MLflow:
- Non-KD + None Pruning
- Non-KD + Single Pruning
- Non-KD + Progressive Pruning
- KD + None Pruning
- KD + Single Pruning
- KD + Progressive Pruning

This script spawns subprocesses to call train_kd.py and prune_and_finetune.py
with Hydra overrides. Adjust the overrides near the bottom for quick changes.
"""

from __future__ import annotations
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
PY = sys.executable


def run(cmd: list[str]) -> None:
    print("\n>>>", " ".join(cmd))
    ret = subprocess.run(cmd)
    if ret.returncode != 0:
        raise SystemExit(ret.returncode)


def main():
    # Common overrides
    base = [PY, "-m", "scripts.train_kd"]

    # Scenario A: Non-KD, None Pruning
    run(base + [
        "train.kd.enabled=false",
        "train.pruning.enabled=false",
        "train.checkpoint_name=student_nonkd_none.pt",
        "experiment_name=NonKD-None",
    ])

    # Scenario B: Non-KD, Single Pruning
    run([PY, "-m", "scripts.prune_and_finetune",
         "train.kd.enabled=false",
         "train.pruning.enabled=true",
         "train.pruning.mode=single",
         "train.checkpoint_name=student_nonkd_none.pt",
         "experiment_name=NonKD-Single",
    ])

    # Scenario C: Non-KD, Progressive Pruning
    run([PY, "-m", "scripts.prune_and_finetune",
         "train.kd.enabled=false",
         "train.pruning.enabled=true",
         "train.pruning.mode=progressive",
         "train.pruning.progressive_steps=3",
         "train.checkpoint_name=student_nonkd_none.pt",
         "experiment_name=NonKD-Progressive",
    ])

    # Scenario D: KD, None Pruning
    run(base + [
        "train.kd.enabled=true",
        "train.kd.alpha=0.7",
        "train.kd.temperature=4.0",
        "train.checkpoint_name=student_kd_none.pt",
        "experiment_name=KD-None",
    ])

    # Scenario E: KD, Single Pruning
    run([PY, "-m", "scripts.prune_and_finetune",
         "train.kd.enabled=true",
         "train.kd.alpha=0.7",
         "train.kd.temperature=4.0",
         "train.pruning.enabled=true",
         "train.pruning.mode=single",
         "train.checkpoint_name=student_kd_none.pt",
         "experiment_name=KD-Single",
    ])

    # Scenario F: KD, Progressive Pruning
    run([PY, "-m", "scripts.prune_and_finetune",
         "train.kd.enabled=true",
         "train.kd.alpha=0.7",
         "train.kd.temperature=4.0",
         "train.pruning.enabled=true",
         "train.pruning.mode=progressive",
         "train.pruning.progressive_steps=3",
         "train.checkpoint_name=student_kd_none.pt",
         "experiment_name=KD-Progressive",
    ])


if __name__ == "__main__":
    main()
