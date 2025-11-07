"""Instrument detection (tagging) wrappers with a registry-based backend system.

This module abstracts open pretrained audio tagging models (PaSST/AST/PANNs).
For P0, we register stub backends; real integrations can replace them later
without changing the pipeline code.
"""
from __future__ import annotations
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
from .utils import Registry

@dataclass
class DetectionConfig:
    """Configuration for a detection backend.

    Attributes:
        name: Backend name (e.g., 'passt', 'ast', 'panns').
        classes: List of instrument labels to report.
        threshold: Score threshold for positive detection.
        source: 'hub' or 'local' to load weights.
        checkpoint_path: Local checkpoint path when source='local'.
        mel_params: Optional dict of Log-Mel parameters for backends that need it.
    """

    name: str
    classes: List[str]
    threshold: float = 0.4
    source: str = "hub"
    checkpoint_path: Optional[str] = None
    mel_params: Optional[Dict] = None


# Global registry for detector backends
DETECTOR_REGISTRY = Registry("detector")


@DETECTOR_REGISTRY.register("stub")
def build_stub_detector():
    """Return a simple detector callable that outputs zeros for requested classes.

    Use this as a placeholder until a real backend is integrated.
    """

    def _detect(waveform: torch.Tensor, sr: int, cfg: DetectionConfig) -> Dict[str, float]:
        _ = (waveform, sr)
        return {label: 0.0 for label in cfg.classes}

    return _detect


# Aliases to stub for now; replace with real integrations later.
@DETECTOR_REGISTRY.register("passt")
def build_passt_detector():
    return build_stub_detector()


@DETECTOR_REGISTRY.register("ast")
def build_ast_detector():
    return build_stub_detector()


@DETECTOR_REGISTRY.register("panns")
def build_panns_detector():
    return build_stub_detector()


def build_detector(name: str = "passt"):
    """Build a detector callable from registry by backend name."""
    factory = DETECTOR_REGISTRY.get(name)
    return factory()


def detect_instruments(
    waveform: torch.Tensor,
    sr: int,
    classes: List[str],
    name: str = "passt",
    threshold: float = 0.4,
    source: str = "hub",
    checkpoint_path: Optional[str] = None,
    mel_params: Optional[Dict] = None,
) -> Dict[str, float]:
    """Run instrument detection.

    Args:
        waveform: (1, T) or (C, T) float32 tensor.
        sr: Sample rate.
        classes: Instrument labels to report.
        name: Backend name.
        threshold: Threshold for filtering (returned scores unaffected; filtering is caller's choice).
        source: Where to load weights ('hub' or 'local').
        checkpoint_path: Local checkpoint path when source='local'.
        mel_params: Optional dict of Log-Mel params (n_fft, hop_length, n_mels, ...).

    Returns:
        Dict mapping instrument -> score (0..1). Caller can filter by threshold.
    """
    cfg = DetectionConfig(
        name=name,
        classes=classes,
        threshold=threshold,
        source=source,
        checkpoint_path=checkpoint_path,
        mel_params=mel_params,
    )
    detector = build_detector(name=name)
    scores = detector(waveform, sr, cfg)
    # Ensure all requested classes exist
    for label in classes:
        scores.setdefault(label, 0.0)
    return scores
