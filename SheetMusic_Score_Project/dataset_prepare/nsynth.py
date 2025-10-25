"""
Utilities to materialize small, class-balanced subsets of audio datasets from
Hugging Face Datasets into a local folder-of-wavs layout.

This module focuses on NSynth instrument families as a practical, package-based
alternative to direct URL downloads. The saved structure is:
  <root>/nsynth/<label>/<index>.wav
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Iterable

import numpy as np
import torch
import torchaudio
from datasets import load_dataset, DatasetInfo
from dataset_prepare import register_preparer

# Canonical NSynth instrument families (string labels)
HF_NSYNTH_FAMILIES = [
    "bass", "brass", "flute", "guitar", "keyboard",
    "mallet", "organ", "reed", "string", "synth_lead", "vocal"
]


def _ensure_dir(p: Path) -> None:
    """
    Create the directory if it does not exist (mkdir -p).

    Args:
        p: Target directory path to create.
    """
    p.mkdir(parents=True, exist_ok=True)


def _to_tensor(audio_array: np.ndarray) -> torch.Tensor:
    """
    Convert a 1D numpy array waveform to a float32 torch.Tensor of shape (1, T).

    Args:
        audio_array: Input waveform as a numpy array, shape (T,) or (1, T).

    Returns:
        A mono torch tensor with shape (1, T) and dtype float32.
    """
    if audio_array.ndim == 1:
        audio_array = audio_array[None, ...]
    return torch.from_numpy(audio_array.astype(np.float32))


def _try_hf_stream(
    dataset_id: str,
    split: str,
    dataset_config_name: Optional[str] = None,
):
    """
    Attempt to open an HF dataset in streaming mode.

    Returns:
        (iterable, features) on success, or (None, None) on failure.
    """
    try:
        ds = load_dataset(dataset_id, dataset_config_name, split=split, streaming=True)
        return ds, getattr(ds, "features", None)
    except Exception as e:
        print(f"[HF] streaming load failed for '{dataset_id}': {e}")
        return None, None


def _resolve_nsynth_stream(split: str) -> Tuple[Optional[Iterable], Optional[Dict]]:
    """
    Try multiple dataset IDs for NSynth and return a streaming iterable when possible.

    Order:
      1) 'google/nsynth' (preferred HF Hub ID)
      2) 'nsynth'       (alias on HF, if present)

    Note:
      - 'tensorflow_datasets' backend is not used here for streaming, because
        TFDS streaming via HF is typically not supported. It would trigger
        large local downloads, which we want to avoid by default.
    """
    for ds_id in ("google/nsynth", "nsynth"):
        ds, features = _try_hf_stream(ds_id, split=split, dataset_config_name=None)
        if ds is not None:
            return ds, features
    return None, None


def prepare_hf_audio_subset(
    dataset_name: str,
    split: str,
    out_dir: Path,
    label_column: str,
    audio_column: str = "audio",
    keep_labels: Optional[Sequence[str]] = None,
    max_per_label: int = 100,
    sampling_rate_override: Optional[int] = None,
    dataset_config_name: Optional[str] = None,
) -> Path:
    """
    Save a small, label-balanced subset of a Hugging Face audio dataset to local WAV files.

    This function streams examples from HF Datasets (no full download), filters by label,
    optionally resamples audio, and writes each example as a WAV into:
      out_dir/<label>/<index>.wav

    Args:
        dataset_name: HF dataset identifier (e.g., "google/nsynth" or "nsynth").
        split: Dataset split to read (e.g., "train", "valid", "test"). NSynth uses "valid".
        out_dir: Output directory where WAV files will be saved.
        label_column: Column name for class labels (string or int). If int, we try to map
            it to human-readable names via dataset features (when available).
        audio_column: Column name holding audio data (default: "audio").
        keep_labels: Optional list of label names to keep. If None, keep all labels found.
        max_per_label: Maximum number of examples to save per label.
        sampling_rate_override: If set, resample audio to this sampling rate before saving.
        dataset_config_name: Optional dataset config (HF "name") if the dataset requires one.

    Returns:
        Path to the output directory containing the saved WAV files.

    Notes:
        - Streaming mode is used to avoid large downloads. If streaming is not available
          for the given dataset name, this function will raise an error to prevent
          accidental bulk downloads.
        - The function stops early once every requested label reaches max_per_label.
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # Try to open the dataset in streaming mode.
    if dataset_name in ("nsynth", "google/nsynth"):
        ds_stream, features = _resolve_nsynth_stream(split=split)
    else:
        ds_stream, features = _try_hf_stream(dataset_name, split=split, dataset_config_name=dataset_config_name)

    if ds_stream is None:
        raise RuntimeError(
            f"Streaming not available for dataset='{dataset_name}'. "
            "Try 'google/nsynth' or 'nsynth', and ensure the split is correct (e.g., 'valid')."
        )

    keep_set = set(keep_labels) if keep_labels is not None else None
    saved_counts: Dict[str, int] = {}

    for ex in ds_stream:
        # HF audio schema: dict with "array" and "sampling_rate".
        audio = ex.get(audio_column)
        if not isinstance(audio, dict) or "array" not in audio:
            continue

        # Resolve label to string, mapping int -> names when possible.
        label_value = ex.get(label_column)
        if label_value is None:
            continue
        if isinstance(label_value, int) and features is not None:
            try:
                names = features[label_column].names  # type: ignore[index]
                label_str = names[label_value]
            except Exception:
                label_str = str(label_value)
        else:
            label_str = str(label_value)

        if keep_set is not None and label_str not in keep_set:
            continue

        n = saved_counts.get(label_str, 0)
        if n >= max_per_label:
            continue

        # Convert to tensor and optionally resample.
        wav = _to_tensor(np.asarray(audio["array"]))
        sr = int(audio["sampling_rate"])
        if sampling_rate_override and sampling_rate_override != sr:
            resampler = torchaudio.transforms.Resample(sr, sampling_rate_override)
            wav = resampler(wav)
            sr = sampling_rate_override

        # Save WAV: out_dir/<label>/<index>.wav
        label_dir = out_dir / label_str
        _ensure_dir(label_dir)
        out_path = label_dir / f"{n:06d}.wav"
        torchaudio.save(out_path.as_posix(), wav, sr)
        saved_counts[label_str] = n + 1

        # Early exit once all requested labels reach the quota.
        if keep_set is not None:
            done = all(saved_counts.get(lbl, 0) >= max_per_label for lbl in keep_set)
            if done:
                break

    print(f"[HF] Saved counts per label: {saved_counts} -> {out_dir}")
    return out_dir


@register_preparer("nsynth")
def prepare_nsynth_subset(
    root: Path,
    split: str = "train",
    families: Optional[Sequence[str]] = None,
    max_per_family: int = 100,
    target_sr: int = 16000,
) -> Path:
    """
    Materialize a small NSynth subset (by instrument family) as local WAV files.

    This is a convenience wrapper over prepare_hf_audio_subset that targets NSynth's
    instrument families. It creates:
      <root>/nsynth/<family>/<index>.wav

    Args:
        root: Root directory under which the "nsynth" folder will be created.
        split: NSynth split to read ("train", "valid", or "test").
        families: List of family names to keep. If None, a small starter set is used
            (e.g., ["guitar", "flute", "keyboard"]).
        max_per_family: Maximum number of clips to save per family.
        target_sr: Sampling rate to resample audio to before saving (project standard).

    Returns:
        Path to the created "<root>/nsynth" directory.
    """
    if families is None:
        families = ["guitar", "flute", "keyboard"]  # small starter set (≤1GB)
    out = Path(root) / "nsynth"

    return prepare_hf_audio_subset(
        dataset_name="google/nsynth",   # try official HF Hub ID first
        dataset_config_name=None,
        split=split,                    # 'train' | 'valid' | 'test'
        out_dir=out,
        label_column="instrument_family",
        audio_column="audio",
        keep_labels=families,
        max_per_label=max_per_family,
        sampling_rate_override=target_sr,
    )