"""Instrument detection (tagging) wrappers with a registry-based backend system.

This module abstracts open pretrained audio tagging models (PaSST/AST/PANNs).
For P0, we register stub backends; real integrations can replace them later
without changing the pipeline code.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

import torch
import importlib
import warnings
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
_PASST_CACHE: Dict[str, torch.nn.Module] = {}
_PASST_PROCESSOR_CACHE: Dict[str, Callable] = {}


def _load_passt(model_id: str = "Jing-Ma/passt-s-f128-p16-s10-ap476"):
    """Lazy-load PaSST model and processor from Hugging Face.

    Caches model and processor for subsequent calls. Returns (model, processor).
    Raises RuntimeError with guidance if transformers not installed or load fails.
    """
    if model_id in _PASST_CACHE and model_id in _PASST_PROCESSOR_CACHE:
        return _PASST_CACHE[model_id], _PASST_PROCESSOR_CACHE[model_id]
    try:
        transformers = importlib.import_module("transformers")
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "transformers is required for PaSST detection. Install with: pip install transformers"
        ) from e
    try:
        processor = transformers.AutoProcessor.from_pretrained(model_id)
        model = transformers.AutoModelForAudioClassification.from_pretrained(model_id)
        model.eval()
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to load PaSST model '{model_id}': {e}") from e
    _PASST_CACHE[model_id] = model
    _PASST_PROCESSOR_CACHE[model_id] = processor
    return model, processor


def _map_user_label_to_audioset(label: str) -> List[str]:
    """Map a user-level instrument label to possible AudioSet class substrings.

    We do substring matching over model.config.id2label values. Returns a list of
    candidate label strings to aggregate. If no mapping found, returns [] and caller
    will produce 0 score for that user label.
    """
    l = label.lower()
    if l == "piano":
        return ["piano"]
    if l == "guitar":
        return ["guitar"]  # acoustic/electric aggregated by substring
    if l == "bass":
        return ["bass"]
    if l == "drums":
        return ["drum", "drums"]
    if l == "vocals" or l == "voice":
        return ["singing", "vocal", "voice"]
    return [l]


@DETECTOR_REGISTRY.register("passt")
def build_passt_detector():
    """Return a detector callable using PaSST (Hugging Face)."""

    def _detect(waveform: torch.Tensor, sr: int, cfg: DetectionConfig) -> Dict[str, float]:
        try:
            model, processor = _load_passt()
        except Exception as e:
            warnings.warn(str(e))
            return {label: 0.0 for label in cfg.classes}

        # Ensure mono float tensor
        wav = waveform.detach().cpu().float()
        if wav.dim() == 2:
            wav = wav.mean(dim=0)
        wav_np = wav.numpy()

        # Processor handles resampling internally if needed (depends on model config)
        inputs = processor(wav_np, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits.squeeze(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy()

        # HuggingFace audio classification models expose an id2label dict or list.
        raw_id2label = getattr(model.config, "id2label", None)
        if raw_id2label is None:
            warnings.warn("PaSST model has no id2label; returning zeros.")
            return {label: 0.0 for label in cfg.classes}
        # Normalize into a list indexed by class id
        if isinstance(raw_id2label, dict):
            id2label_list = [raw_id2label[i] for i in range(len(raw_id2label))]
        else:
            id2label_list = list(raw_id2label)
        label_scores: Dict[str, float] = {str(id2label_list[i]).lower(): float(probs[i]) for i in range(len(probs))}

        results: Dict[str, float] = {}
        for user_label in cfg.classes:
            substrings = _map_user_label_to_audioset(user_label)
            matched = [p for lab, p in label_scores.items() if any(sub in lab for sub in substrings)]
            score = float(max(matched) if matched else 0.0)
            results[user_label] = score
        # Optionally apply threshold filtering (still return full dict for caller)
        return results

    return _detect


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
    # Optionally, callers may want filtered positives; we keep full scores for now
    # but clamp to [0,1] and sanity-check types
    scores = {k: float(max(0.0, min(1.0, float(v)))) for k, v in scores.items()}
    return scores
