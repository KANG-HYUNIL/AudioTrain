"""Transcription backends: Basic Pitch, Onsets-and-Frames, and simple monophonic pitch.

P0 provides signatures and minimal stubs. Replace with real integrations later.
"""
from __future__ import annotations
from typing import Dict, Optional
from pathlib import Path

import torch


class MidiTrack:
    """Lightweight MIDI track placeholder.

    This is a struct-like container until we integrate a real MIDI library.
    """
    def __init__(self, instrument: str, notes: Optional[list] = None) -> None:
        self.instrument = instrument
        self.notes = notes or []  # list of (onset_s, offset_s, pitch, velocity)


class MidiBundle:
    """Container for multiple tracks to be packaged later."""
    def __init__(self) -> None:
        self.tracks: Dict[str, MidiTrack] = {}


def transcribe_basic_pitch(audio_path: str | Path) -> MidiTrack:
    """Transcribe using Basic Pitch (placeholder).

    Args:
        audio_path: Path to stem audio file.

    Returns:
        MidiTrack with extracted notes (empty in P0).
    """
    return MidiTrack(instrument="unknown", notes=[])


def transcribe_onsets_frames(waveform: torch.Tensor, sr: int) -> MidiTrack:
    """Transcribe using Onsets-and-Frames (placeholder)."""
    _ = (waveform, sr)
    return MidiTrack(instrument="unknown", notes=[])


def pick_transcriber(instrument: str, default: str = "basic_pitch"):
    """Choose transcriber callable based on instrument label.

    Returns:
        A callable that produces a MidiTrack.
    """
    if default == "basic_pitch":
        return transcribe_basic_pitch
    return transcribe_onsets_frames
