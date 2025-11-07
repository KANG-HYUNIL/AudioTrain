"""Packaging utilities: merge per-instrument tracks and export to files.

Uses pretty_midi for MIDI export when available; falls back to placeholder.
MusicXML export remains a lightweight placeholder to be replaced by music21.
"""
from __future__ import annotations
from typing import Dict, Optional
from pathlib import Path

from .transcription import MidiBundle, MidiTrack
import importlib


def _gm_program_for(instrument: str) -> int:
    """Return a General MIDI program index (0-127) for a given instrument label.

    This is a heuristic mapping; adjust as needed.
    """
    name = instrument.lower()
    if "piano" in name:
        return 0  # Acoustic Grand Piano
    if "guitar" in name:
        return 24  # Acoustic Guitar (nylon)
    if "bass" in name:
        return 32  # Acoustic Bass (fingered)
    if any(k in name for k in ["violin", "strings", "cello", "viola"]):
        return 48  # Strings Ensemble 1
    if "organ" in name:
        return 16  # Drawbar Organ
    if "synth" in name:
        return 80  # Synth Lead (square)
    if "voice" in name or "vocal" in name or "choir" in name:
        return 52  # Choir Aahs
    return 0


def merge_tracks(tracks: Dict[str, MidiTrack], tempo_bpm: Optional[float] = None, metadata: Optional[dict] = None) -> MidiBundle:
    """Merge individual instrument tracks into a multi-track bundle.

    Args:
        tracks: Mapping instrument -> MidiTrack
        tempo_bpm: Optional tempo to embed later
        metadata: Optional free-form metadata dictionary

    Returns:
        MidiBundle with tracks merged (here, just a container copy).
    """
    bundle = MidiBundle()
    bundle.tracks = dict(tracks)
    return bundle


def export_midi(bundle: MidiBundle, output_path: str | Path) -> str:
    """Export a MidiBundle to a MIDI file using pretty_midi when available.

    Falls back to creating a placeholder file if pretty_midi is not installed.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    try:
        pm = importlib.import_module("pretty_midi")
    except Exception:
        out.write_bytes(b"Placeholder MIDI content. Install pretty_midi for real export.")
        return str(out)

    midi = pm.PrettyMIDI()
    for name, track in bundle.tracks.items():
        is_drum = "drum" in name.lower()
        program = 0 if is_drum else _gm_program_for(name)
        inst = pm.Instrument(program=program, is_drum=is_drum, name=name)
        for onset, offset, pitch, velocity in track.notes:
            onset_s = max(0.0, float(onset))
            offset_s = max(onset_s + 1e-3, float(offset))
            note = pm.Note(velocity=int(max(1, min(127, velocity))), pitch=int(pitch), start=onset_s, end=offset_s)
            inst.notes.append(note)
        midi.instruments.append(inst)
    midi.write(str(out))
    return str(out)


def export_musicxml(bundle: MidiBundle, output_path: str | Path) -> str:
    """Export to MusicXML using music21 (placeholder)."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("<!-- Placeholder MusicXML. Replace with music21 export. -->\n")
    return str(out)
