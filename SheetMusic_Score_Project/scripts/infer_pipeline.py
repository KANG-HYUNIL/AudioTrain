"""Hydra entrypoint for the audio-to-score inference pipeline.

Usage (PowerShell / cross-platform):
  python -m scripts.infer_pipeline pipeline.io.input_path=path/to/audio.wav pipeline.io.output_dir=./outputs

On Colab, prefer using the notebook that installs dependencies and calls this module.
"""
from __future__ import annotations

from omegaconf import DictConfig, OmegaConf
import hydra

import logging

from audio2score.pipeline import run_pipeline

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("==== Composed Config ====")
    log.info(OmegaConf.to_yaml(cfg))

    # Expect new group cfg.pipeline (see configs/pipeline/base.yaml)
    input_path = str(cfg.pipeline.io.input_path)
    if not input_path:
        raise SystemExit("Set pipeline.io.input_path=<audio file> via CLI or config.")

    artifacts, stats = run_pipeline(input_path, cfg)
    log.info(f"Artifacts: {artifacts}")
    log.info(f"Stats: {stats}")


if __name__ == "__main__":
    main()
