"""
Collate utilities to build mini-batches with a fixed time dimension.

This module provides a collate function designed for spectrogram-like features
(e.g., Log-Mel) with shape (C, F, T). It crops or pads the time axis to a
user-specified target length so that a batch can be stacked into a single
(B, C, F, T_target) tensor.

Typical usage:
    loader = DataLoader(
        dataset,
        batch_size=..., shuffle=True, num_workers=..., pin_memory=True,
        collate_fn=lambda batch: collate_fixed_length(batch, target_frames=512)
    )

Notes:
- This function expects each batch element to be a tuple of
  (features, label_idx) or (features, label_idx, path).
- "features" is a torch.Tensor with shape (C, F, T_i), where T_i can vary.
- It performs random cropping when T_i > target_frames and right-padding when
  T_i < target_frames.
- Keep random cropping to train-time only; for validation/testing, use center
  crop (random_crop=False) for determinism.
"""

from __future__ import annotations
from typing import List, Tuple, Union

import torch

BatchItem = Union[Tuple[torch.Tensor, int], Tuple[torch.Tensor, int, str]]


def _crop_or_pad_time(x: torch.Tensor, t_target: int, *, random_crop: bool = True, pad_value: float = 0.0) -> torch.Tensor:
    """
    Ensure the time dimension of `x` matches `t_target` by cropping or padding.

    Args:
        x: Input tensor of shape (C, F, T).
        t_target: Desired length for the time axis.
        random_crop: Use random crop when T > t_target. If False, use center crop.
        pad_value: Value used for right padding when T < t_target.

    Returns:
        Tensor of shape (C, F, t_target).
    """
    assert x.dim() == 3, f"Expected 3D tensor (C, F, T), got shape={tuple(x.shape)}"
    C, F, T = x.shape

    if T == t_target:
        return x

    if T > t_target:
        # Crop
        if random_crop:
            start = torch.randint(0, T - t_target + 1, (1,)).item()
        else:
            start = max(0, (T - t_target) // 2)
        return x[:, :, start : start + t_target]

    # Pad on the right: (C, F, T) -> (C, F, t_target)
    pad_T = t_target - T
    pad_spec = (0, pad_T, 0, 0, 0, 0)  # (pad_T_right, pad_T_left=0) per dim order (W, H, C)
    # For 3D tensor, torch.nn.functional.pad expects pad in reverse order of dims
    # We'll use torch.pad via F.pad with (last two dims per pair): (T, F, C)
    import torch.nn.functional as F
    return F.pad(x, pad=pad_spec, mode="constant", value=pad_value)


def collate_fixed_length(
    batch: List[BatchItem],
    target_frames: int,
    *,
    random_crop: bool = True,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    Collate function that stacks variable-length spectrograms into a fixed-length batch.

    Args:
        batch: A list of (features, label_idx) or (features, label_idx, path).
               `features` must be shaped as (C, F, T_i).
        target_frames: The desired time dimension length for the stacked batch.
        random_crop: Apply random cropping when T_i > target_frames. If False,
                     use center cropping (useful for validation/testing).
        pad_value: Constant value used for right-padding when T_i < target_frames.

    Returns:
        features: Tensor of shape (B, C, F, target_frames)
        labels:   Tensor of shape (B,)
        paths:    List[str] of length B (empty strings if paths were not provided)
    """
    feats_list: List[torch.Tensor] = []
    labels_list: List[int] = []
    paths_list: List[str] = []

    for item in batch:
        if len(item) == 3:
            feats, label, path = item  # type: ignore[misc]
        else:
            feats, label = item  # type: ignore[misc]
            path = ""
        # Expect (C, F, T) or (1, F, T)
        if feats.dim() == 4 and feats.shape[0] == 1:
            # Some pipelines return (B=1, C, F, T). Squeeze batch dim.
            feats = feats.squeeze(0)
        assert feats.dim() == 3, f"features must be (C, F, T); got {tuple(feats.shape)}"
        feats = _crop_or_pad_time(feats, target_frames, random_crop=random_crop, pad_value=pad_value)
        feats_list.append(feats)
        labels_list.append(int(label))
        paths_list.append(path)

    features = torch.stack(feats_list, dim=0)  # (B, C, F, T)
    labels = torch.tensor(labels_list, dtype=torch.long)
    return features, labels, paths_list
