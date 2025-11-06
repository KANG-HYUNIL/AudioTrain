"""Instrument detection (tagging) wrappers.

This module provides a thin abstraction over open pretrained audio tagging models
(e.g., PaSST/AST/PANNs). For P0, implementations can be stubbed or progressively
added. The interface is detection-model agnostic.
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import torch


class DetectionConfig:
    """Lightweight detection config.

    Attributes:
        name: Model backend name (e.g., 'passt', 'ast', 'panns').
        classes: List of instrument labels to report.
        threshold: Score threshold for positive detection.
    """

    def __init__(self, name: str, classes: List[str], threshold: float = 0.4) -> None:
        self.name = name
        self.classes = classes
        self.threshold = float(threshold)


def build_detector(name: str = "passt"):
    """Build a detector callable given a backend name.

    Returns:
        A callable like detect(waveform: Tensor, sr: int, cfg: DetectionConfig) -> Dict[str, float]
    """

    def _stub_detect(waveform: torch.Tensor, sr: int, cfg: DetectionConfig) -> Dict[str, float]:
        """Stub detector: returns empty or uniform low scores.

        Replace with real model integration (PaSST/AST/PANNs) in subsequent steps.
        """
        _ = (waveform, sr)
        return {label: 0.0 for label in cfg.classes}

    # Future: route to concrete backend loaders
    return _stub_detect


def detect_instruments(
    waveform: torch.Tensor,
    sr: int,
    classes: List[str],
    name: str = "passt",
    threshold: float = 0.4,
) -> Dict[str, float]:
    """Run instrument detection.

    Args:
        waveform: (1, T) or (C, T) float32 tensor.
        sr: Sample rate.
        classes: Instrument labels to report.
        name: Backend name.
        threshold: Threshold for filtering (returned scores unaffected; filtering is caller's choice).

    Returns:
        Dict mapping instrument -> score (0..1). Caller can filter by threshold.
    """
    cfg = DetectionConfig(name=name, classes=classes, threshold=threshold)
    detector = build_detector(name)
    scores = detector(waveform, sr, cfg)
    # Ensure all requested classes exist
    for label in classes:
        scores.setdefault(label, 0.0)
    return scores
