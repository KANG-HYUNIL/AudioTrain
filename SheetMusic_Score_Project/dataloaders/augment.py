"""
Simple SpecAugment utilities for spectrogram-like inputs.

This module implements a minimal SpecAugment with time and frequency masking for
features shaped as (C, F, T). It is intended to be composed after a mel
spectrogram extractor. Use it for training only; disable for validation/testing.

References:
- SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition
  (Park et al., 2019)
"""

from __future__ import annotations
from typing import Optional

import torch


class SpecAugment:
    """
    Apply time and frequency masking to a spectrogram tensor.

    The input is expected to be a float tensor of shape (C, F, T). For each call,
    the augmenter samples random mask widths and positions and sets the masked
    regions to zero.

    Args:
        max_time_mask_pct: Maximum fraction of the time axis that a single time mask can cover.
        max_freq_mask_pct: Maximum fraction of the frequency axis that a single freq mask can cover.
        num_time_masks: Number of time masks to apply per call.
        num_freq_masks: Number of frequency masks to apply per call.
        mask_value: Constant value to fill in the masked regions (default: 0.0).
    """

    def __init__(
        self,
        *,
        max_time_mask_pct: float = 0.1,
        max_freq_mask_pct: float = 0.15,
        num_time_masks: int = 2,
        num_freq_masks: int = 2,
        mask_value: float = 0.0,
    ) -> None:
        self.max_time_mask_pct = float(max_time_mask_pct)
        self.max_freq_mask_pct = float(max_freq_mask_pct)
        self.num_time_masks = int(num_time_masks)
        self.num_freq_masks = int(num_freq_masks)
        self.mask_value = float(mask_value)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply in-place-like masking (works on a cloned tensor to avoid side effects).

        Args:
            x: Input tensor of shape (C, F, T), typically log-mel features.

        Returns:
            Masked tensor of the same shape.
        """
        assert x.dim() == 3, f"SpecAugment expects (C, F, T), got {tuple(x.shape)}"
        C, F, T = x.shape
        out = x.clone()

        # Frequency masks
        max_f = max(1, int(F * self.max_freq_mask_pct))
        for _ in range(max(0, self.num_freq_masks)):
            f = int(torch.randint(0, max_f + 1, (1,)).item())
            if f == 0:
                continue
            high_f0 = int(max(1, F - f + 1))
            f0 = int(torch.randint(0, high_f0, (1,)).item())
            out[:, f0 : f0 + f, :] = self.mask_value

        # Time masks
        max_t = max(1, int(T * self.max_time_mask_pct))
        for _ in range(max(0, self.num_time_masks)):
            t = int(torch.randint(0, max_t + 1, (1,)).item())
            if t == 0:
                continue
            high_t0 = int(max(1, T - t + 1))
            t0 = int(torch.randint(0, high_t0, (1,)).item())
            out[:, :, t0 : t0 + t] = self.mask_value

        return out


class ComposeMelAndAug:
    """
    Compose a mel feature extractor and an optional SpecAug augmentor.

    The first callable should convert (waveform, src_sr) to a spectrogram-like tensor.
    The optional second callable takes that tensor and returns an augmented tensor.
    """

    def __init__(self, mel_transform, specaug: Optional[SpecAugment] = None) -> None:
        self.mel_transform = mel_transform
        self.specaug = specaug

    def __call__(self, waveform: torch.Tensor, src_sr: int) -> torch.Tensor:
        feats = self.mel_transform(waveform, src_sr=src_sr)
        # mel_transform returns (C=1, F, T); SpecAug expects (C, F, T)
        if self.specaug is not None:
            if feats.dim() == 4 and feats.shape[0] == 1:
                feats = feats.squeeze(0)
            feats = self.specaug(feats)
        return feats
"""
Audio augmentation utilities: SpecAug (time/freq mask), RIR/codec, gain/noise, mixup.
TODO: Implement using torch-audiomentations / torchaudio.
"""

from typing import Optional


class Augmenter:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        # TODO: initialize augmentation pipeline

    def __call__(self, features):
        # TODO: apply augmentations conditionally
        return features
