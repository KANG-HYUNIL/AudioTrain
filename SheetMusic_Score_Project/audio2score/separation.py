"""Source separation wrappers.

Default backend: Demucs (HTDemucs). For P0 we expose a function signature and a
stub that simply returns the input as a single 'mix' stem.
"""
from __future__ import annotations
from typing import Dict, Union
from pathlib import Path

import torch


def separate_demucs(
    input_audio: Union[str, Path, torch.Tensor],
    sample_rate: int,
    model_name: str = "htdemucs",
) -> Dict[str, torch.Tensor]:
    """Separate audio into stems using Demucs.

    Args:
        input_audio: Path to audio file or waveform tensor (1, T) / (C, T).
        sample_rate: Input sample rate.
        model_name: Demucs model name.

    Returns:
        Dict of stem_name -> waveform tensor (C, T).

    Notes:
        This is a placeholder for P0. Replace with Demucs inference in later step.
    """
    if isinstance(input_audio, (str, Path)):
        # In a later implementation, call Demucs CLI/API and load the resulting stems.
        # For P0 stub: return empty dict to signal separation not applied.
        return {}
    # If tensor provided, return a single 'mix' stem as pass-through
    wav = input_audio.detach().clone()
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    return {"mix": wav}


def map_stems_to_instruments(stems: Dict[str, torch.Tensor]) -> Dict[str, str]:
    """Map stem names to instrument groups.

    Args:
        stems: Dict of stem name -> waveform.

    Returns:
        Dict of stem name -> instrument group label.

    Notes:
        Implement mapping based on configs.pipeline.separator.stem_map.
    """
    return {name: name for name in stems.keys()}
